"""Reporting helpers for benchmark outputs and plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _apply_period_ticks(ax, values, label="Year"):
    """Format x-axis ticks for yearly or quarterly periods."""
    if not values:
        ax.set_xlabel(label)
        return

    has_fraction = any(abs(v - int(v)) > 1e-6 for v in values)
    if not has_fraction:
        ax.set_xlabel(label)
        return

    ticks = sorted(values)
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


def compute_confidence_intervals(results_df: pd.DataFrame, ci=0.95, group_by=None) -> pd.DataFrame:
    """Compute 95% confidence intervals for each metric by method (and optionally other grouping)."""
    if group_by is None:
        group_by = ["method"]
    
    metrics = [col for col in results_df.columns if col not in ["seed", "method", "maintenance"]]
    
    summary_data = []
    for group_vals in results_df.groupby(group_by, sort=True).groups:
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        group_data = results_df
        for i, col in enumerate(group_by):
            group_data = group_data[group_data[col] == group_vals[i]]
        
        row = {col: group_vals[i] for i, col in enumerate(group_by)}
        
        for metric in metrics:
            values = group_data[metric].values
            if len(values) > 0:
                mean = np.mean(values)
                stderr = stats.sem(values)
                ci_range = stderr * stats.t.ppf((1 + ci) / 2, len(values) - 1)
                
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = np.std(values)
                row[f"{metric}_ci_lower"] = mean - ci_range
                row[f"{metric}_ci_upper"] = mean + ci_range
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def statistical_tests_vs_baseline(results_df: pd.DataFrame) -> pd.DataFrame:
    """Run t-tests comparing each method vs baseline."""
    metrics = [col for col in results_df.columns if col not in ["seed", "method", "maintenance", "year"]]
    baseline_data = results_df[results_df["method"] == "baseline"]
    
    test_results = []
    for method in sorted(results_df["method"].unique()):
        if method == "baseline":
            continue
        
        method_data = results_df[results_df["method"] == method]
        row = {"method": method}
        
        for metric in metrics:
            baseline_vals = baseline_data[metric].values
            method_vals = method_data[metric].values
            
            t_stat, p_val = stats.ttest_ind(method_vals, baseline_vals)
            row[f"{metric}_tstat"] = t_stat
            row[f"{metric}_pval"] = p_val
            row[f"{metric}_significant"] = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        
        test_results.append(row)
    
    return pd.DataFrame(test_results)


def plot_static_comparison(summary_ci: pd.DataFrame, output_dir: Path):
    """Plot static method comparison as bar charts."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['dp_gap', 'eo_gap', 'accuracy', 'auc']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
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
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Method")
        ax.grid(True, axis="y")
    
    plt.tight_layout()
    plot_path = output_dir / "static_comparison.png"
    _save_figure(fig, plot_path)
    plt.close(fig)
    print(f"✓ Saved static comparison plot to {plot_path}")


def plot_temporal_comparison_by_year(results_by_year_df: pd.DataFrame, output_dir: Path):
    """Plot temporal metrics with error bands (CI) by year and method."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['dp_gap', 'eo_gap', 'accuracy', 'auc']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        grouped = results_by_year_df.groupby(['year', 'method'])[metric].agg(['mean', 'std', 'count'])
        grouped['ci'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
        
        colors = sns.color_palette("muted", n_colors=len(results_by_year_df['method'].unique()))
        
        for idx, method in enumerate(sorted(results_by_year_df['method'].unique())):
            method_data = grouped.loc[grouped.index.get_level_values('method') == method]
            years = [y for y, _ in method_data.index]
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
        
        _apply_period_ticks(ax, years, label="Year")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Temporal {metric.upper()} by Method (95% CI)")
        ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.grid(True)
        
        plot_path = output_dir / f"temporal_{metric}_by_method.png"
        _save_figure(fig, plot_path)
        plt.close(fig)
    
    print(f"✓ Saved temporal comparison plots")


def plot_temporal_metrics(results_by_year_df: pd.DataFrame, output_dir: Path):
    """Plot temporal metrics aggregated by year (legacy function)."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["dp_gap", "eo_gap", "accuracy", "auc"]

    for metric in metrics:
        pivot = (
            results_by_year_df.groupby(["year", "method"])[metric]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        for method in sorted(pivot["method"].unique()):
            data_m = pivot[pivot["method"] == method]
            ax.plot(data_m["year"], data_m[metric], linewidth=2, label=method)

        _apply_period_ticks(ax, data_m["year"].tolist(), label="Year")
        ax.set_ylabel(metric)
        ax.set_title(f"Temporal {metric} by year")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        plot_path = output_dir / f"temporal_{metric}.png"
        _save_figure(fig, plot_path)
        plt.close(fig)


def plot_original_vs_updated(results_by_year_df, output_dir):
    """Plot Original Model (no-retrain) vs Updated Model (retrain) for each method."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "maintenance" not in results_by_year_df.columns:
        return
    
    maintenance_opts = results_by_year_df["maintenance"].unique()
    if len(maintenance_opts) < 2:
        return
    
    metrics = ["dp_gap", "eo_gap", "accuracy", "auc"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    axes = axes.flatten()
    
    colors = {"no-retrain": "#6E6E6E", "retrain": "#2C7FB8"}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        grouped = results_by_year_df.groupby(["year", "maintenance"])[metric].agg(["mean", "std", "count"])
        grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

        for maintenance in sorted(maintenance_opts):
            data_m = grouped.loc[grouped.index.get_level_values("maintenance") == maintenance]
            years = [y for y, _ in data_m.index]
            means = data_m["mean"].values
            cis = data_m["ci"].values

            line_style = "-" if maintenance == "retrain" else "--"
            label = "Updated model" if maintenance == "retrain" else "Original model"

            ax.plot(years, means, linewidth=2.0, linestyle=line_style, color=colors[maintenance], label=label)
            ax.fill_between(years, means - cis, means + cis, color=colors[maintenance], alpha=0.15)

        # Baseline initial performance (first period) as a reference line
        if years:
            baseline_value = means[0]
            ax.axhline(baseline_value, color="#333333", linestyle=":", linewidth=1.2, label="Initial performance")

        _apply_period_ticks(ax, years, label="Year")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.grid(True)
        ax.legend(fontsize=9, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    
    fig.suptitle("Original vs Updated Models", y=1.02)
    fig.tight_layout()
    
    plot_path = output_dir / "temporal_original_vs_updated.png"
    _save_figure(fig, plot_path)
    print(f"✓ Saved original vs updated plot to {plot_path}")
    plt.close(fig)
