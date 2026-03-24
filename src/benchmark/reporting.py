"""Reporting helpers for benchmark outputs and plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.benchmark.metrics import METRIC_NAMES, METRIC_LABELS


def _t_ci_half_width(grouped_df: pd.DataFrame, ci: float = 0.95) -> pd.Series:
    """Compute per-row CI half-width using the t-distribution.

    Using the t-distribution (rather than the normal approximation z=1.96) is
    important for the small sample sizes typical in these benchmarks (n ≈ 10–20
    seeds).  For n=10 seeds, t_{9}(0.975) ≈ 2.26 vs z = 1.96, giving CIs that
    are ~13% wider – a meaningful difference at this scale.

    Args:
        grouped_df: DataFrame with ``count`` and ``std`` columns (as produced
            by ``.agg(['mean', 'std', 'count'])``).
        ci: Confidence level (default 0.95).

    Returns:
        Series of CI half-widths aligned to ``grouped_df``'s index.
    """
    n = grouped_df["count"].values.astype(float)
    s = grouped_df["std"].values
    df_dof = np.maximum(n - 1, 1)
    t_crit = stats.t.ppf((1 + ci) / 2, df_dof)
    return pd.Series(t_crit * s / np.sqrt(n), index=grouped_df.index)


# Mapping for attribute names to readable labels
ATTRIBUTE_NAME_LABELS = {
    "SEX": "SEX",
    "RAC1P": "RACE"
}

GAP_METRICS = {"dp_gap", "eo_gap", "oe_gap"}

# Mapping for sensitive attribute values to readable labels
ATTRIBUTE_VALUE_LABELS = {
    "SEX": {0: "Female", 1: "Male"},
    "RAC1P": {
        0: "Non-White",
        1: "White",
        2: "Black",
        3: "Native American",
        4: "Alaska Native",
        5: "Native Hawaiian",
        6: "Asian",
        7: "Other Pacific Islander",
        8: "Other",
        9: "Two or More Races"
    }
}


def _apply_paper_style():
    """Apply a clean, paper-like plotting style."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        },
    )


def _save_figure(fig, path: Path):
    """Save both PNG and PDF for publication-ready output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _plot_output_path(output_dir: Path, category: str, filename: str) -> Path:
    """Build a plot path under output_dir/plots/<category>/<filename>."""
    return output_dir / "plots" / category / filename


def _apply_period_ticks(ax, values, label="Year"):
    """Format x-axis ticks for yearly or quarterly periods."""
    if not values:
        ax.set_xlabel(label)
        return

    sorted_values = sorted(set(values))
    has_fraction = any(abs(v - int(v)) > 1e-6 for v in values)
    if not has_fraction:
        ax.set_xlabel(label)
        if len(sorted_values) >= 2:
            ax.set_xlim(sorted_values[0], sorted_values[-1])
        ax.margins(x=0)
        return

    ticks = sorted_values
    labels = []
    if len(ticks) > 16:
        ticks = ticks[::4]
    elif len(ticks) > 10:
        ticks = ticks[::2]
    for v in ticks:
        year = int(np.floor(v))
        quarter = int(round((v - year) * 4)) + 1
        labels.append(f"{year}Q{quarter}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Quarter")
    if len(sorted_values) >= 2:
        ax.set_xlim(sorted_values[0], sorted_values[-1])
    ax.margins(x=0)


def flatten_summary_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for col in summary_df.columns:
        if isinstance(col, tuple):
            if col[0] == "method":
                columns.append("method")
            else:
                columns.append(f"{col[0]}_{col[1]}")
        else:
            columns.append(col)
    summary_df.columns = columns
    return summary_df


def _available_summary_metrics(summary_ci: pd.DataFrame) -> list[str]:
    """Get metrics available in summary CI table, preserving registry order."""
    metrics = [metric for metric in METRIC_NAMES if f"{metric}_mean" in summary_ci.columns]
    if metrics:
        return metrics

    detected = []
    for col in summary_ci.columns:
        if col.endswith("_mean"):
            detected.append(col[:-5])
    return sorted(set(detected))


def _available_results_metrics(results_df: pd.DataFrame) -> list[str]:
    """Get numeric metrics available in raw yearly results, preserving registry order."""
    metrics = [metric for metric in METRIC_NAMES if metric in results_df.columns]
    if metrics:
        return metrics

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    return [col for col in numeric_cols if col not in ["seed", "year"]]


def compute_confidence_intervals(results_df: pd.DataFrame, ci=0.95, group_by=None) -> pd.DataFrame:
    """Compute 95% confidence intervals for each metric by method (and optionally other grouping)."""
    if group_by is None:
        group_by = ["method"]
    
    # If sensitive_attribute is present and not in group_by, add it
    if "sensitive_attribute" in results_df.columns and "sensitive_attribute" not in group_by:
        group_by = group_by + ["sensitive_attribute"]
    
    # Filter for numeric columns only
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    metrics = [col for col in numeric_cols if col not in ["seed", "year"]]
    
    summary_data = []
    for group_vals in results_df.groupby(group_by, sort=True).groups:
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        group_data = results_df
        for i, col in enumerate(group_by):
            group_data = group_data[group_data[col] == group_vals[i]]
        
        row = {col: group_vals[i] for i, col in enumerate(group_by)}
        
        for metric in metrics:
            values = pd.to_numeric(group_data[metric], errors="coerce").dropna().values
            if len(values) == 0:
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            if len(values) < 2:
                ci_range = 0.0
            else:
                stderr = stats.sem(values, nan_policy="omit")
                if np.isnan(stderr):
                    ci_range = 0.0
                else:
                    ci_range = float(stderr * stats.t.ppf((1 + ci) / 2, len(values) - 1))

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_lower"] = mean - ci_range
            row[f"{metric}_ci_upper"] = mean + ci_range
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def statistical_tests_vs_baseline(results_df: pd.DataFrame, stratify_by: list[str] | None = None) -> pd.DataFrame:
    """Run t-tests comparing each method vs baseline, optionally stratified by context columns."""
    if stratify_by is None:
        stratify_by = []
    stratify_by = [col for col in stratify_by if col in results_df.columns]

    # Filter for numeric columns only, excluding metadata and string columns
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    metrics = [col for col in numeric_cols if col not in ["seed", "year"]]
    
    test_results = []
    if stratify_by:
        grouped_contexts = results_df.groupby(stratify_by, sort=True)
        context_iter = [
            (ctx_vals if isinstance(ctx_vals, tuple) else (ctx_vals,), ctx_df)
            for ctx_vals, ctx_df in grouped_contexts
        ]
    else:
        context_iter = [(tuple(), results_df)]

    for context_vals, context_df in context_iter:
        baseline_data = context_df[context_df["method"] == "baseline"]
        if baseline_data.empty:
            continue

        context_dict = {col: context_vals[i] for i, col in enumerate(stratify_by)}

        for method in sorted(context_df["method"].unique()):
            if method == "baseline":
                continue

            method_data = context_df[context_df["method"] == method]
            if method_data.empty:
                continue

            row = {**context_dict, "method": method}

            for metric in metrics:
                baseline_vals = pd.to_numeric(baseline_data[metric], errors="coerce").dropna().values
                method_vals = pd.to_numeric(method_data[metric], errors="coerce").dropna().values

                if len(baseline_vals) < 2 or len(method_vals) < 2:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_ind(method_vals, baseline_vals, equal_var=False, nan_policy="omit")

                row[f"{metric}_tstat"] = t_stat
                row[f"{metric}_pval"] = p_val
                if np.isnan(p_val):
                    row[f"{metric}_significant"] = ""
                else:
                    row[f"{metric}_significant"] = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))

            test_results.append(row)
    
    return pd.DataFrame(test_results)


def plot_static_comparison(summary_ci: pd.DataFrame, output_dir: Path):
    """Plot static method comparison as bar charts."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = _available_summary_metrics(summary_ci)
    if not metrics:
        return

    n_cols = 2 if len(metrics) > 1 else 1
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), dpi=300)
    axes = np.atleast_1d(axes).flatten()
    
    colors = sns.color_palette("muted", n_colors=len(summary_ci))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        mean_col = f"{metric}_mean"
        ci_lower_col = f"{metric}_ci_lower"
        ci_upper_col = f"{metric}_ci_upper"
        
        if mean_col not in summary_ci.columns:
            continue
        
        x_pos = np.arange(len(summary_ci))
        means = summary_ci[mean_col].values
        errors_lower = (means - summary_ci[ci_lower_col].values)
        errors_upper = (summary_ci[ci_upper_col].values - means)
        
        ax.bar(
            x_pos,
            means,
            yerr=[errors_lower, errors_upper],
            capsize=3,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in summary_ci['method']], 
                           rotation=45, ha='right')
        metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Method")
        ax.grid(True, axis="y")

    for ax in axes[len(metrics):]:
        ax.remove()
    
    plt.tight_layout()
    plot_path = _plot_output_path(output_dir, "static_comparison", "static_comparison.png")
    _save_figure(fig, plot_path)
    plt.close(fig)
    print(f"✓ Saved static comparison plot to {plot_path}")


def plot_temporal_comparison_by_year(results_by_year_df: pd.DataFrame, output_dir: Path):
    """Plot temporal metrics with error bands (CI) by year and method - generates per-attribute plots."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to ALL rows only (overall metrics, not subgroup-specific)
    if "sensitive_attribute_value" in results_by_year_df.columns:
        results_by_year_df = results_by_year_df[results_by_year_df["sensitive_attribute_value"] == "ALL"]
    
    metrics = _available_results_metrics(results_by_year_df)
    if not metrics:
        return
    
    # Generate separate plots for each sensitive attribute (cannot average gap metrics across attributes)
    attributes = sorted(results_by_year_df["sensitive_attribute"].unique()) if "sensitive_attribute" in results_by_year_df.columns else [None]
    
    for attr_name in attributes:
        if attr_name is not None:
            attr_data = results_by_year_df[results_by_year_df["sensitive_attribute"] == attr_name]
            attr_suffix = f"_{attr_name.lower()}"
            attr_title_suffix = f" - {attr_name}"
        else:
            attr_data = results_by_year_df
            attr_suffix = ""
            attr_title_suffix = ""
        
        if attr_data.empty:
            continue
    
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

            group_cols = ["year", "method"]
            has_maintenance = "maintenance" in attr_data.columns
            if has_maintenance:
                group_cols.append("maintenance")

            grouped = attr_data.groupby(group_cols)[metric].agg(['mean', 'std', 'count'])
            grouped['ci'] = _t_ci_half_width(grouped)

            methods = sorted(attr_data['method'].unique())
            colors = sns.color_palette("muted", n_colors=len(methods))
            line_styles = {"no-retrain": "--", "retrain": "-"}

            all_years = sorted(attr_data["year"].unique())

            for idx, method in enumerate(methods):
                if has_maintenance:
                    maint_opts = sorted(attr_data["maintenance"].dropna().unique())
                    for maintenance in maint_opts:
                        method_data = grouped[
                            (grouped.index.get_level_values("method") == method)
                            & (grouped.index.get_level_values("maintenance") == maintenance)
                        ]
                        if method_data.empty:
                            continue
                        years = method_data.index.get_level_values("year").to_numpy(dtype=float)
                        means = method_data['mean'].values
                        cis = method_data['ci'].values

                        label = f"{method.replace('_', ' ').title()} ({maintenance.replace('-', ' ')})"
                        ax.plot(
                            years,
                            means,
                            linewidth=2.0,
                            linestyle=line_styles.get(maintenance, "-"),
                            label=label,
                            color=colors[idx],
                        )
                        ax.fill_between(years, means - cis, means + cis, color=colors[idx], alpha=0.12)
                else:
                    method_data = grouped[grouped.index.get_level_values("method") == method]
                    if method_data.empty:
                        continue
                    years = method_data.index.get_level_values("year").to_numpy(dtype=float)
                    means = method_data['mean'].values
                    cis = method_data['ci'].values

                    ax.plot(
                        years,
                        means,
                        linewidth=2.0,
                        label=method.replace('_', ' ').title(),
                        color=colors[idx],
                    )
                    ax.fill_between(years, means - cis, means + cis, color=colors[idx], alpha=0.15)

            _apply_period_ticks(ax, all_years, label="Year")
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            ax.set_ylabel(metric_label)
            ax.set_title(f"Temporal {metric_label} by Method (95% CI){attr_title_suffix}")
            ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
            ax.grid(True)
            
            plot_path = _plot_output_path(output_dir, "temporal_by_method", f"temporal_{metric}_by_method{attr_suffix}.png")
            _save_figure(fig, plot_path)
            plt.close(fig)
    
    print(f"✓ Saved temporal comparison plots")


def plot_temporal_metrics(results_by_year_df: pd.DataFrame, output_dir: Path):
    """Plot temporal metrics aggregated by year (legacy function - generates per-attribute plots)."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _available_results_metrics(results_by_year_df)
    if not metrics:
        return

    # Filter to ALL rows only (overall metrics, not subgroup-specific)
    if "sensitive_attribute_value" in results_by_year_df.columns:
        results_by_year_df = results_by_year_df[results_by_year_df["sensitive_attribute_value"] == "ALL"]
    
    # Generate separate plots for each sensitive attribute (cannot average gap metrics across attributes)
    attributes = sorted(results_by_year_df["sensitive_attribute"].unique()) if "sensitive_attribute" in results_by_year_df.columns else [None]
    
    for attr_name in attributes:
        if attr_name is not None:
            attr_data = results_by_year_df[results_by_year_df["sensitive_attribute"] == attr_name]
            attr_suffix = f"_{attr_name.lower()}"
        else:
            attr_data = results_by_year_df
            attr_suffix = ""
        
        if attr_data.empty:
            continue
    
        for metric in metrics:
            group_cols = ["year", "method"]
            has_maintenance = "maintenance" in attr_data.columns
            if has_maintenance:
                group_cols.append("maintenance")

            pivot = attr_data.groupby(group_cols)[metric].mean().reset_index()
            all_years = sorted(pivot["year"].unique())

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
            for method in sorted(pivot["method"].unique()):
                if has_maintenance:
                    for maintenance in sorted(pivot["maintenance"].dropna().unique()):
                        data_m = pivot[(pivot["method"] == method) & (pivot["maintenance"] == maintenance)]
                        if data_m.empty:
                            continue
                        label = f"{method} ({maintenance})"
                        line_style = "--" if maintenance == "no-retrain" else "-"
                        ax.plot(data_m["year"], data_m[metric], linewidth=2, linestyle=line_style, label=label)
                else:
                    data_m = pivot[pivot["method"] == method]
                    if data_m.empty:
                        continue
                    ax.plot(data_m["year"], data_m[metric], linewidth=2, label=method)

            _apply_period_ticks(ax, all_years, label="Year")
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            ax.set_ylabel(metric_label)
            title_suffix = f" ({attr_name})" if attr_name else ""
            ax.set_title(f"Temporal {metric_label} by year{title_suffix}")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend()
            fig.tight_layout()

            plot_path = _plot_output_path(output_dir, "temporal_metrics", f"temporal_{metric}{attr_suffix}.png")
            _save_figure(fig, plot_path)
            plt.close(fig)


def plot_original_vs_updated(results_by_year_df, output_dir):
    """Plot Original Model (no-retrain) vs Updated Model (retrain) - generates per-attribute plots."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "maintenance" not in results_by_year_df.columns:
        return
    
    # Filter to ALL rows only (overall metrics, not subgroup-specific)
    if "sensitive_attribute_value" in results_by_year_df.columns:
        results_by_year_df = results_by_year_df[results_by_year_df["sensitive_attribute_value"] == "ALL"]
    
    maintenance_opts = results_by_year_df["maintenance"].unique()
    if len(maintenance_opts) < 2:
        return
    
    metrics = _available_results_metrics(results_by_year_df)
    if not metrics:
        return

    has_multiple_attributes = (
        "sensitive_attribute" in results_by_year_df.columns
        and results_by_year_df["sensitive_attribute"].nunique() > 1
    )
    general_metrics = [metric for metric in metrics if metric not in GAP_METRICS] if has_multiple_attributes else metrics
    if not general_metrics:
        return

    def _build_original_vs_updated_figure(plot_data: pd.DataFrame, title_suffix: str, metrics_to_plot: list[str]):
        n_cols = 2 if len(metrics_to_plot) > 1 else 1
        n_rows = int(np.ceil(len(metrics_to_plot) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), dpi=200)
        axes = np.atleast_1d(axes).flatten()

        colors = {"no-retrain": "#6E6E6E", "retrain": "#2C7FB8"}
        years_sorted = sorted(plot_data["year"].unique())

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            grouped = plot_data.groupby(["year", "maintenance"])[metric].agg(["mean", "std", "count"])
            grouped["ci"] = _t_ci_half_width(grouped)

            for maintenance in sorted(maintenance_opts):
                data_m = grouped.loc[grouped.index.get_level_values("maintenance") == maintenance]
                if data_m.empty:
                    continue

                years = [y for y, _ in data_m.index]
                means = data_m["mean"].values
                cis = np.nan_to_num(data_m["ci"].values, nan=0.0)

                line_style = "-" if maintenance == "retrain" else "--"
                label = "Updated model" if maintenance == "retrain" else "Original model"

                ax.plot(years, means, linewidth=2.0, linestyle=line_style, color=colors[maintenance], label=label)
                ax.fill_between(years, means - cis, means + cis, color=colors[maintenance], alpha=0.15)

            if years_sorted:
                first_period = years_sorted[0]
                baseline_rows = plot_data[
                    (plot_data["maintenance"] == "no-retrain")
                    & (plot_data["year"] == first_period)
                ]
                if not baseline_rows.empty:
                    baseline_value = baseline_rows[metric].mean()
                    ax.axhline(
                        baseline_value,
                        color="#333333",
                        linestyle=":",
                        linewidth=1.2,
                        label="Initial performance",
                    )

            _apply_period_ticks(ax, years_sorted, label="Year")
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            preferred_order = ["Initial performance", "Original model", "Updated model"]
            ordered = [
                (handles[labels.index(label)], label)
                for label in preferred_order
                if label in labels
            ]
            if ordered:
                ordered_handles, ordered_labels = zip(*ordered)
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    fontsize=9,
                    ncol=3,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                )

        for ax in axes[len(metrics_to_plot):]:
            ax.remove()

        fig.suptitle(f"Original vs Updated Models{title_suffix}", y=1.02)
        fig.tight_layout()
        return fig

    # General view (legacy behavior): aggregate across all available sensitive attributes
    general_fig = _build_original_vs_updated_figure(
        results_by_year_df,
        " - General",
        general_metrics,
    )
    general_path = _plot_output_path(output_dir, "original_vs_updated_general", "original_vs_updated_general.png")
    _save_figure(general_fig, general_path)
    combined_general_path = _plot_output_path(output_dir, "original_vs_updated_combined", "original_vs_updated_general.png")
    _save_figure(general_fig, combined_general_path)
    plt.close(general_fig)

    # Generate separate plots for each sensitive attribute (cannot average gap metrics across attributes)
    attributes = sorted(results_by_year_df["sensitive_attribute"].unique()) if "sensitive_attribute" in results_by_year_df.columns else [None]
    
    for attr_name in attributes:
        if attr_name is not None:
            attr_data = results_by_year_df[results_by_year_df["sensitive_attribute"] == attr_name]
            attr_suffix = f"_{attr_name.lower()}"
            attr_title_suffix = f" - {attr_name}"
        else:
            attr_data = results_by_year_df
            attr_suffix = ""
            attr_title_suffix = ""
        
        if attr_data.empty:
            continue

        fig = _build_original_vs_updated_figure(attr_data, attr_title_suffix, metrics)
        
        combined_path = _plot_output_path(output_dir, "original_vs_updated_combined", f"original_vs_updated{attr_suffix}.png")
        _save_figure(fig, combined_path)
        plt.close(fig)
    
    print(f"✓ Saved original vs updated plots (general + per attribute)")


def plot_original_vs_updated_by_attribute(results_by_year_df, output_dir):
    """Plot Original vs Updated Models stratified by sensitive attribute values.
    
    Creates one subplot per sensitive attribute, with different lines for each
    attribute value (e.g., Female vs Male for SEX).
    """
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "sensitive_attribute" not in results_by_year_df.columns:
        return
    
    if "sensitive_attribute_value" not in results_by_year_df.columns:
        return
    
    if "maintenance" not in results_by_year_df.columns:
        return
    
    maintenance_opts = results_by_year_df["maintenance"].unique()
    if len(maintenance_opts) < 2:
        return
    
    metrics = _available_results_metrics(results_by_year_df)
    if not metrics:
        return
    
    # Get unique attributes (not attribute-value pairs)
    unique_attributes = sorted(results_by_year_df["sensitive_attribute"].unique())
    if not unique_attributes:
        return
    
    n_attrs = len(unique_attributes)
    n_cols = min(2, n_attrs)
    n_rows = int(np.ceil(n_attrs / n_cols))
    
    # Color palette for different attribute values (max 10 colors)
    value_colors = plt.cm.tab10.colors
    maintenance_colors = {"no-retrain": "#6E6E6E", "retrain": "#2C7FB8"}
    years_sorted = sorted(results_by_year_df["year"].unique())
    
    for metric in metrics:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), dpi=200)
        axes = np.atleast_1d(axes).flatten()
        
        for idx, attr_name in enumerate(unique_attributes):
            ax = axes[idx]
            plotted_any = False
            metric_is_gap = metric in GAP_METRICS
            
            # Get all values for this attribute
            attr_data = results_by_year_df[results_by_year_df["sensitive_attribute"] == attr_name]
            if attr_data.empty:
                continue

            if metric_is_gap:
                # Gap metrics are defined across groups, so use overall per-attribute rows (ALL)
                overall_data = attr_data[attr_data["sensitive_attribute_value"].astype(str) == "ALL"]
                if not overall_data.empty:
                    grouped = overall_data.groupby(["year", "maintenance"])[metric].agg(["mean", "std", "count"])
                    grouped["ci"] = _t_ci_half_width(grouped)

                    for maintenance in sorted(maintenance_opts):
                        data_m = grouped.loc[grouped.index.get_level_values("maintenance") == maintenance]
                        if data_m.empty:
                            continue

                        years = [y for y, _ in data_m.index]
                        means = data_m["mean"].values
                        cis = data_m["ci"].values

                        valid_mask = ~np.isnan(means)
                        if not np.any(valid_mask):
                            continue

                        years = np.array(years)[valid_mask]
                        means = means[valid_mask]
                        cis = np.nan_to_num(cis[valid_mask], nan=0.0)

                        line_style = "-" if maintenance == "retrain" else "--"
                        label = "Updated model" if maintenance == "retrain" else "Original model"
                        color = maintenance_colors.get(maintenance, "#2C7FB8")

                        ax.plot(years, means, linewidth=2.0, linestyle=line_style, color=color, label=label)
                        ax.fill_between(years, means - cis, means + cis, color=color, alpha=0.15)
                        plotted_any = True
            else:
                # Non-gap metrics: keep subgroup-level lines, excluding ALL rows
                subgroup_data = attr_data[attr_data["sensitive_attribute_value"].astype(str) != "ALL"]
                unique_values = sorted(subgroup_data["sensitive_attribute_value"].unique(), key=lambda value: str(value))

                for val_idx, attr_value in enumerate(unique_values):
                    value_data = subgroup_data[subgroup_data["sensitive_attribute_value"] == attr_value]
                    if value_data.empty:
                        continue

                    value_label = attr_value
                    if attr_name in ATTRIBUTE_VALUE_LABELS:
                        value_label = ATTRIBUTE_VALUE_LABELS[attr_name].get(attr_value, attr_value)

                    base_color = value_colors[val_idx % len(value_colors)]

                    grouped = value_data.groupby(["year", "maintenance"])[metric].agg(["mean", "std", "count"])
                    grouped["ci"] = _t_ci_half_width(grouped)

                    for maintenance in sorted(maintenance_opts):
                        data_m = grouped.loc[grouped.index.get_level_values("maintenance") == maintenance]
                        if data_m.empty:
                            continue

                        years = [y for y, _ in data_m.index]
                        means = data_m["mean"].values
                        cis = data_m["ci"].values

                        valid_mask = ~np.isnan(means)
                        if not np.any(valid_mask):
                            continue

                        years = np.array(years)[valid_mask]
                        means = means[valid_mask]
                        cis = np.nan_to_num(cis[valid_mask], nan=0.0)

                        line_style = "-" if maintenance == "retrain" else "--"
                        maintenance_label = "Updated" if maintenance == "retrain" else "Original"
                        label = f"{value_label} - {maintenance_label}"

                        ax.plot(years, means, linewidth=2.0, linestyle=line_style, color=base_color, label=label)
                        ax.fill_between(years, means - cis, means + cis, color=base_color, alpha=0.15)
                        plotted_any = True
            
            # Baseline initial performance from first period (averaged across all values)
            if years_sorted:
                first_period = years_sorted[0]
                baseline_rows = attr_data[
                    (attr_data["maintenance"] == "no-retrain")
                    & (attr_data["year"] == first_period)
                ]
                if not baseline_rows.empty:
                    baseline_value = baseline_rows[metric].mean()
                    if not np.isnan(baseline_value):
                        ax.axhline(
                            baseline_value,
                            color="#333333",
                            linestyle=":",
                            linewidth=1.2,
                            label="Initial performance",
                        )
                        plotted_any = True
            
            _apply_period_ticks(ax, years_sorted, label="Year")
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            ax.set_ylabel(metric_label)
            
            # Use readable attribute name (RAC1P -> RACE)
            attr_display_name = ATTRIBUTE_NAME_LABELS.get(attr_name, attr_name)
            ax.set_title(f"{attr_display_name}")
            ax.grid(True)
            
            if plotted_any:
                # Organize legend in 3 columns:
                # Row 1: Value1-Original | Initial performance | Value2-Original
                # Row 2: Value1-Updated  | (empty)             | Value2-Updated
                handles, labels = ax.get_legend_handles_labels()

                if metric_is_gap:
                    ax.legend(
                        handles,
                        labels,
                        fontsize=8,
                        ncol=min(len(labels), 3),
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),
                    )
                    continue

                # Remove duplicates by converting to dict (last occurrence wins)
                # Then reconstruct in order of first appearance
                seen_order = []
                label_to_handle = {}
                for hdl, lbl in zip(handles, labels):
                    if lbl not in label_to_handle:
                        seen_order.append(lbl)
                    label_to_handle[lbl] = hdl

                # Extract unique attribute values from labels
                value_labels = []
                for lbl in seen_order:
                    if " - " in lbl and lbl != "Initial performance":
                        val_lbl = lbl.split(" - ")[0]
                        if val_lbl not in value_labels:
                            value_labels.append(val_lbl)

                # Reorder for 3-column layout: positions [0,1,2] = row1, [3,4,5] = row2
                if len(value_labels) >= 2:
                    ordered_handles = []
                    ordered_labels = []

                    val1_orig = f"{value_labels[0]} - Original"
                    val2_orig = f"{value_labels[1]} - Original"
                    val1_upd = f"{value_labels[0]} - Updated"
                    val2_upd = f"{value_labels[1]} - Updated"

                    if val1_orig in label_to_handle:
                        ordered_handles.append(label_to_handle[val1_orig])
                        ordered_labels.append(val1_orig)
                    if "Initial performance" in label_to_handle:
                        ordered_handles.append(label_to_handle["Initial performance"])
                        ordered_labels.append("Initial performance")
                    if val2_orig in label_to_handle:
                        ordered_handles.append(label_to_handle[val2_orig])
                        ordered_labels.append(val2_orig)
                    if val1_upd in label_to_handle:
                        ordered_handles.append(label_to_handle[val1_upd])
                        ordered_labels.append(val1_upd)

                    from matplotlib.patches import Rectangle
                    invisible_handle = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
                    ordered_handles.append(invisible_handle)
                    ordered_labels.append("")

                    if val2_upd in label_to_handle:
                        ordered_handles.append(label_to_handle[val2_upd])
                        ordered_labels.append(val2_upd)

                    ax.legend(
                        ordered_handles,
                        ordered_labels,
                        fontsize=8,
                        ncol=3,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),
                    )
                else:
                    ax.legend(
                        handles,
                        labels,
                        fontsize=8,
                        ncol=min(len(labels), 5),
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A for subgroup-level plot\n(metric undefined or all missing)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#666666",
                )
        
        for ax in axes[n_attrs:]:
            ax.remove()
        
        metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        fig.suptitle(f"{metric_label} - Original vs Updated Models by Attribute", y=1.02)
        fig.tight_layout()
        
        plot_path = _plot_output_path(output_dir, "original_vs_updated_by_attribute", f"temporal_original_vs_updated_by_attribute_{metric}.png")
        _save_figure(fig, plot_path)
        plt.close(fig)
    
    print(f"✓ Saved original vs updated by attribute plots")
