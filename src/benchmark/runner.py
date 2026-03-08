"""Benchmark runner implementation (static and temporal)."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.benchmark.progress import ProgressTracker, ProgressCalculator
from src.benchmark.data import (
    load_folktables,
    load_folktables_by_period,
    stratified_split,
)
from src.benchmark.methods import (
    make_model,
    kamiran_calders_weights,
    choose_thresholds_equalized_odds,
    train_with_lagrangian,
)
from src.benchmark.metrics import compute_metrics, METRIC_NAMES, METRIC_LABELS
from src.benchmark.tables.initial_performance import _format_ci
from src.benchmark.tables.summary_by_attribute import generate_summary_tables_by_attribute
from src.benchmark.reporting import (
    flatten_summary_columns,
    plot_temporal_metrics,
    plot_static_comparison,
    plot_temporal_comparison_by_year,
    plot_original_vs_updated,
    plot_original_vs_updated_by_attribute,
    compute_confidence_intervals,
    statistical_tests_vs_baseline,
)


def _compute_stratified_metrics(y_true, y_pred, y_proba, A, sensitive_attr_name):
    """Compute metrics stratified by sensitive attribute values.
    
    Returns a list of dicts, one per unique value of A, with structure:
    {
        'sensitive_attribute': sensitive_attr_name,
        'sensitive_attribute_value': attr_value,
        **metrics
    }
    
    Note: Fairness gaps (dp_gap, eo_gap, oe_gap) will be NaN for stratified subgroups
    since they measure disparities between groups, not within a single group.
    """
    import warnings
    
    results = []
    unique_values = sorted(np.unique(A))
    
    for attr_val in unique_values:
        mask = A == attr_val
        if mask.sum() == 0:
            continue
        
        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]
        y_proba_subset = y_proba[mask]
        A_subset = A[mask]
        
        # Suppress warnings about fairness gaps on single-group subsets
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            warnings.filterwarnings('ignore', message='invalid value encountered')
            metrics = compute_metrics(y_true_subset, y_pred_subset, y_proba_subset, A_subset)
        
        results.append({
            'sensitive_attribute': sensitive_attr_name,
            'sensitive_attribute_value': int(attr_val),
            **metrics
        })
    
    return results


def _generate_initial_performance_table(
    summary_ci: pd.DataFrame,
    task: str,
    output_dir: Path,
    maintenance: str | None = None,
    sensitive_attribute: str | None = None,
):
    """Generate initial performance table (metrics by method with CI) for paper."""
    table_source = summary_ci.copy()
    if "task" in table_source.columns:
        table_source = table_source[table_source["task"] == task]
    if maintenance is not None and "maintenance" in table_source.columns:
        table_source = table_source[table_source["maintenance"] == maintenance]
    if sensitive_attribute is not None and "sensitive_attribute" in table_source.columns:
        table_source = table_source[table_source["sensitive_attribute"] == sensitive_attribute]

    if table_source.empty:
        return None

    metrics = [metric for metric in METRIC_NAMES if f"{metric}_mean" in summary_ci.columns]
    if not metrics:
        return None
    
    rows = []
    methods_sorted = sorted(table_source["method"].unique())
    for metric in metrics:
        row = {"Metric": METRIC_LABELS.get(metric, metric.replace("_", " ").title())}
        
        for method in methods_sorted:
            method_data = table_source[table_source["method"] == method]
            
            if not method_data.empty:
                val = _format_ci(method_data.iloc[0], metric)
            else:
                val = "-"
            
            col_name = method.replace("_", " ").title()
            row[col_name] = val
        
        rows.append(row)
    
    table_df = pd.DataFrame(rows)
    
    # Save as LaTeX
    latex_table = table_df.to_latex(index=False, escape=False)
    table_path = output_dir / f"initial_performance_{task}.tex"
    table_path.write_text(latex_table)
    return table_path


def _train_method_on_data(method, X_train, y_train, A_train, X_val, y_val, A_val, seed, threshold_grid, config):
    """Train a single fairness method. Returns model and thresholds if applicable."""
    if method == "baseline":
        model = make_model(seed)
        model.fit(X_train, y_train)
        return model, None
    
    elif method == "reweighing":
        weights = kamiran_calders_weights(y_train, A_train)
        model = make_model(seed)
        model.fit(X_train, y_train, sample_weight=weights)
        return model, None
    
    elif method == "equalized_odds":
        model = make_model(seed)
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        thresholds = choose_thresholds_equalized_odds(y_val, y_val_proba, A_val, grid=threshold_grid)
        return model, thresholds
    
    elif method == "fairness_constraint":
        lag_cfg = config.get("fairness_constraint", {})
        num_iters = int(lag_cfg.get("num_iters", 8))
        lr = float(lag_cfg.get("lr", 0.1))
        model = train_with_lagrangian(X_train, y_train, A_train, seed, num_iters, lr)
        return model, None
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _predict_with_method(method, model, thresholds, X_test, A_test):
    """Get predictions from a trained model."""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    if method == "equalized_odds" and thresholds is not None:
        y_pred = np.where(
            A_test == 0,
            y_proba >= thresholds[0],
            y_proba >= thresholds[1],
        ).astype(int)
    else:
        y_pred = (y_proba >= 0.5).astype(int)
    
    return y_pred, y_proba


def _append_run_history(base_output_dir: Path, history_row: dict):
    history_path = base_output_dir / "run_history.csv"
    history_df = pd.DataFrame([history_row])

    if history_path.exists():
        history_df.to_csv(history_path, mode="a", header=False, index=False)
    else:
        history_df.to_csv(history_path, index=False)

    return history_path


def run_benchmark(config_path: str):
    start_time = time.perf_counter()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    exp_cfg = config["experiment"]
    methods = config["methods"]
    split = config.get("split", [0.6, 0.1, 0.3])
    seeds = config.get("seeds", list(range(20)))
    threshold_grid = config.get("threshold_grid", np.linspace(0.05, 0.95, 19).tolist())

    output_cfg = config["output"]
    configured_output_dir = Path(output_cfg["dir"])
    config_stem = Path(config_path).stem

    group_by_config = bool(output_cfg.get("group_by_config", True))
    if group_by_config and configured_output_dir.name != config_stem:
        base_output_dir = configured_output_dir / config_stem
    else:
        base_output_dir = configured_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_format = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    run_id = datetime.now().strftime(timestamp_format)
    timestamped_runs = bool(output_cfg.get("timestamped_runs", True))
    output_dir = base_output_dir / run_id if timestamped_runs else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config snapshot for this run
    config_out = output_dir / "config_resolved.yaml"
    with open(config_out, "w") as f:
        yaml.dump(config, f)

    mode = data_cfg.get("mode", "static")
    compare_outputs = config.get("compare_outputs", [])

    results_by_year = []
    max_samples = data_cfg.get("max_samples")
    max_samples_per_year = data_cfg.get("max_samples_per_year")
    sample_seed = int(config.get("sample_seed", 42))
    
    # Get maintenance strategies (no-retrain, retrain)
    maintenance_strategies = config.get("maintenance_strategies", ["no-retrain"])

    if mode == "temporal":
        train_years = data_cfg["train_years"]
        val_years = data_cfg["val_years"]
        test_years = data_cfg["test_years"]
        frequency = data_cfg.get("frequency", "year")
        
        # Check if model development is from same year (70/10/20 split)
        same_year_split = train_years == val_years

        if same_year_split:
            # Load pooled data and apply 70/10/20 split
            X_pool_df, y_pool, A_pool = load_folktables(
                task=data_cfg["task"],
                states=data_cfg["states"],
                years=train_years,
                sensitive_attribute=data_cfg["sensitive_attribute"],
                max_samples=max_samples,
                random_state=sample_seed,
                keep_sensitive_cols=True,
            )
            # Apply stratified 70/10/20 split for model development
            split_ratio = [0.7, 0.1, 0.2]  # train, val, test
            X_train, X_val, X_test_dev, y_train, y_val, y_test_dev, A_train, A_val, A_test_dev = stratified_split(
                X_pool_df.values, y_pool, A_pool, seed=sample_seed, split=split_ratio
            )
            # Convert back to DataFrames for consistency
            X_train_df = pd.DataFrame(X_train, columns=X_pool_df.columns)
            X_val_df = pd.DataFrame(X_val, columns=X_pool_df.columns)
            X_test_dev_df = pd.DataFrame(X_test_dev, columns=X_pool_df.columns)
        else:
            # Load training data separately (when years differ)
            X_train_df, y_train, A_train = load_folktables(
                task=data_cfg["task"],
                states=data_cfg["states"],
                years=train_years,
                sensitive_attribute=data_cfg["sensitive_attribute"],
                max_samples=max_samples,
                random_state=sample_seed,
                keep_sensitive_cols=True,
            )
            X_val_df, y_val, A_val = load_folktables(
                task=data_cfg["task"],
                states=data_cfg["states"],
                years=val_years,
                sensitive_attribute=data_cfg["sensitive_attribute"],
                max_samples=max_samples,
                random_state=sample_seed + 1,
                keep_sensitive_cols=True,
            )
        
        # Load temporal test data (only years after development year)
        if same_year_split:
            # Exclude development year from temporal test years
            temporal_test_years = sorted([y for y in test_years if y not in train_years])
        else:
            temporal_test_years = sorted(test_years)
            X_test_dev_df, y_test_dev, A_test_dev = None, None, None
        
        if temporal_test_years:
            X_test_df, y_test, A_test = load_folktables(
                task=data_cfg["task"],
                states=data_cfg["states"],
                years=temporal_test_years,
                sensitive_attribute=data_cfg["sensitive_attribute"],
                max_samples=max_samples,
                random_state=sample_seed + 2,
                keep_sensitive_cols=True,
            )
        else:
            # No temporal years beyond development year (unlikely but handle it)
            X_test_df, y_test, A_test = X_test_dev_df, y_test_dev, A_test_dev

        test_by_year, base_columns = load_folktables_by_period(
            task=data_cfg["task"],
            states=data_cfg["states"],
            years=temporal_test_years,  # Use temporal years only (exclude development year)
            sensitive_attribute=data_cfg["sensitive_attribute"],
            frequency=frequency,
            max_samples_per_year=max_samples_per_year,
            random_state=sample_seed + 3,
        )

        # If no temporal years are loaded, infer base columns from development data
        if base_columns is None:
            base_columns = X_train_df.columns

        X_val_df = X_val_df.reindex(columns=base_columns, fill_value=0)
        X_train_df = X_train_df.reindex(columns=base_columns, fill_value=0)

        if X_test_dev_df is not None:
            X_test_dev_df = X_test_dev_df.reindex(columns=base_columns, fill_value=0)

        X_test_df = X_test_df.reindex(columns=base_columns, fill_value=0)

        # (Development year test split kept separate, not included in temporal evaluation)

        # Calculate progress steps
        num_periods = len(test_by_year)
        num_methods = len(methods)
        num_seeds = len(seeds)
        development_year_cutoff = max(int(y) for y in (list(train_years) + list(val_years)))
        retrain_test_periods = [
            period
            for period in sorted(test_by_year.keys())
            if int(np.floor(period)) > development_year_cutoff
        ]
        # Both no-retrain and retrain evaluate on the same temporal periods (excluding development year)
        summary_test_periods = sorted(test_by_year.keys())
        summary_test_periods_set = set(summary_test_periods)

        progress_total = 0
        if "no-retrain" in maintenance_strategies:
            progress_total += num_seeds * num_methods  # aggregated test
            progress_total += num_seeds * num_methods * num_periods  # per-period tests
        if "retrain" in maintenance_strategies:
            progress_total += num_seeds * num_methods * len(retrain_test_periods)
    else:
        X_df, y_all, A_all = load_folktables(
            task=data_cfg["task"],
            states=data_cfg["states"],
            years=data_cfg["years"],
            sensitive_attribute=data_cfg["sensitive_attribute"],
            max_samples=max_samples,
            random_state=sample_seed,
            keep_sensitive_cols=True,
        )
        progress_total = ProgressCalculator.calculate_static_steps(
            num_seeds=len(seeds),
            num_methods=len(methods),
        )

    progress = ProgressTracker(progress_total)

    results = []
    for seed in seeds:
        if mode == "temporal":
            X_train = X_train_df.values
            X_val = X_val_df.values
            X_test = X_test_df.values
            X_test_df_for_attrs = X_test_df.copy()  # Keep original for attribute extraction
        else:
            X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = stratified_split(
                X_df.values, y_all, A_all, seed=seed, split=split
            )
            # For static mode, we need to track X_df indices to match with feature columns
            X_test_df_for_attrs = X_df.iloc[np.arange(len(X_df) - len(X_test), len(X_df))].reset_index(drop=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # === NO-RETRAIN SCENARIO ===
        if "no-retrain" in maintenance_strategies:
            for method in methods:
                model, thresholds = _train_method_on_data(
                    method, X_train, y_train, A_train, X_val, y_val, A_val, seed, threshold_grid, config
                )
                
                # Test on aggregated test set
                y_pred, y_proba = _predict_with_method(method, model, thresholds, X_test, A_test)
                
                # Compute metrics on primary attribute (SEX in config)
                metrics = compute_metrics(y_test, y_pred, y_proba, A_test)
                result_entry = {
                    "seed": seed,
                    "method": method,
                    "task": data_cfg["task"],
                    "sensitive_attribute": data_cfg["sensitive_attribute"],
                    **metrics
                }
                if mode != "temporal":
                    results.append(result_entry)

                temporal_primary_metrics = []
                temporal_race_metrics = []
                
                # Compute metrics on secondary attribute (RAC1P) if available and different from primary
                if mode == "temporal" and "RAC1P" in X_test_df_for_attrs.columns and data_cfg["sensitive_attribute"] != "RAC1P":
                    A_race = (X_test_df_for_attrs["RAC1P"] == 1).astype(int)
                    metrics_race = compute_metrics(y_test, y_pred, y_proba, A_race)
                    result_entry_race = {
                        "seed": seed,
                        "method": method,
                        "task": data_cfg["task"],
                        "sensitive_attribute": "RAC1P",
                        **metrics_race
                    }
                    if mode != "temporal":
                        results.append(result_entry_race)
                
                progress.update(f"seed={seed} method={method} test=all")

                # Test on each year
                if mode == "temporal":
                    for year, (X_year, y_year, A_year) in test_by_year.items():
                        X_year_scaled = scaler.transform(X_year.values)
                        y_year_pred, y_year_proba = _predict_with_method(
                            method, model, thresholds, X_year_scaled, A_year
                        )
                        
                        # Compute metrics overall (for aggregation/tables)
                        metrics_year = compute_metrics(y_year, y_year_pred, y_year_proba, A_year)

                        # Save overall yearly metrics for this attribute (used by gap plots)
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "no-retrain",
                            "year": year,
                            "sensitive_attribute": data_cfg["sensitive_attribute"],
                            "sensitive_attribute_value": "ALL",
                            **metrics_year,
                        })
                        
                        # Compute metrics stratified by sensitive attribute values
                        stratified_metrics = _compute_stratified_metrics(
                            y_year, y_year_pred, y_year_proba, A_year, data_cfg["sensitive_attribute"]
                        )
                        for strat_entry in stratified_metrics:
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "year": year,
                                **strat_entry
                            })

                        if year in summary_test_periods_set:
                            temporal_primary_metrics.append(metrics_year)
                        
                        # Secondary attribute on yearly data
                        if "RAC1P" in X_year.columns and data_cfg["sensitive_attribute"] != "RAC1P":
                            A_year_race = (X_year["RAC1P"] == 1).astype(int)
                            metrics_year_race = compute_metrics(y_year, y_year_pred, y_year_proba, A_year_race)

                            # Save overall yearly metrics for secondary attribute (used by gap plots)
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "year": year,
                                "sensitive_attribute": "RAC1P",
                                "sensitive_attribute_value": "ALL",
                                **metrics_year_race,
                            })
                            
                            # Stratified metrics for secondary attribute
                            stratified_race = _compute_stratified_metrics(
                                y_year, y_year_pred, y_year_proba, A_year_race, "RAC1P"
                            )
                            for strat_entry in stratified_race:
                                results_by_year.append({
                                    "seed": seed,
                                    "method": method,
                                    "task": data_cfg["task"],
                                    "maintenance": "no-retrain",
                                    "year": year,
                                    **strat_entry
                                })

                            if year in summary_test_periods_set:
                                temporal_race_metrics.append(metrics_year_race)
                        
                        progress.update(f"seed={seed} method={method} year={year} no-retrain")

                    if temporal_primary_metrics:
                        metric_names = temporal_primary_metrics[0].keys()
                        aggregated_primary_metrics = {
                            metric_name: float(np.nanmean([metrics_row[metric_name] for metrics_row in temporal_primary_metrics]))
                            for metric_name in metric_names
                        }
                        results.append(
                            {
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "sensitive_attribute": data_cfg["sensitive_attribute"],
                                **aggregated_primary_metrics,
                            }
                        )

                    if temporal_race_metrics:
                        metric_names_race = temporal_race_metrics[0].keys()
                        aggregated_race_metrics = {
                            metric_name: float(np.nanmean([metrics_row[metric_name] for metrics_row in temporal_race_metrics]))
                            for metric_name in metric_names_race
                        }
                        results.append(
                            {
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "sensitive_attribute": "RAC1P",
                                **aggregated_race_metrics,
                            }
                        )

        # === RETRAIN SCENARIO ===
        if "retrain" in maintenance_strategies and mode == "temporal":
            # For each evaluated period, retrain on all historical years strictly before it
            sorted_test_periods = retrain_test_periods
            temporal_eval_years = sorted({int(np.floor(p)) for p in sorted_test_periods})
            retrain_metric_accumulator = {}

            for test_period in sorted_test_periods:
                eval_year = int(np.floor(test_period))

                # Cumulative training history: base development years + prior temporal years
                base_train_years = sorted({int(y) for y in train_years})
                prior_temporal_years = [y for y in temporal_eval_years if y < eval_year]
                train_years_for_retrain = sorted(set(base_train_years + prior_temporal_years))
                if not train_years_for_retrain:
                    train_years_for_retrain = base_train_years

                X_retrain_df, y_retrain, A_retrain = load_folktables(
                    task=data_cfg["task"],
                    states=data_cfg["states"],
                    years=train_years_for_retrain,
                    sensitive_attribute=data_cfg["sensitive_attribute"],
                    max_samples=max_samples,
                    random_state=int(sample_seed + 5 + eval_year),
                    keep_sensitive_cols=True,
                )
                X_retrain_df = X_retrain_df.reindex(columns=base_columns, fill_value=0)

                # Validation history follows the same cumulative policy, without leakage
                base_val_years = sorted({int(y) for y in val_years if int(y) < eval_year})
                val_years_for_retrain = sorted(set(base_val_years + prior_temporal_years))
                if not val_years_for_retrain:
                    val_years_for_retrain = train_years_for_retrain

                X_val_retrain_df, y_val_retrain, A_val_retrain = load_folktables(
                    task=data_cfg["task"],
                    states=data_cfg["states"],
                    years=val_years_for_retrain,
                    sensitive_attribute=data_cfg["sensitive_attribute"],
                    max_samples=max_samples,
                    random_state=int(sample_seed + 6 + eval_year),
                    keep_sensitive_cols=True,
                )
                X_val_retrain_df = X_val_retrain_df.reindex(columns=base_columns, fill_value=0)

                # Scale retrain data
                scaler_retrain = StandardScaler()
                X_retrain = scaler_retrain.fit_transform(X_retrain_df.values)
                X_val_retrain = scaler_retrain.transform(X_val_retrain_df.values)

                for method in methods:
                    model, thresholds = _train_method_on_data(
                        method,
                        X_retrain,
                        y_retrain,
                        A_retrain,
                        X_val_retrain,
                        y_val_retrain,
                        A_val_retrain,
                        seed,
                        threshold_grid,
                        config,
                    )

                    # Test on current period only
                    X_year, y_year, A_year = test_by_year[test_period]
                    X_year_scaled = scaler_retrain.transform(X_year.values)
                    y_year_pred, y_year_proba = _predict_with_method(
                        method, model, thresholds, X_year_scaled, A_year
                    )
                    
                    # Compute metrics overall (for aggregation/tables)
                    metrics_year = compute_metrics(y_year, y_year_pred, y_year_proba, A_year)

                    # Save overall yearly metrics for this attribute (used by gap plots)
                    results_by_year.append({
                        "seed": seed,
                        "method": method,
                        "task": data_cfg["task"],
                        "maintenance": "retrain",
                        "year": test_period,
                        "sensitive_attribute": data_cfg["sensitive_attribute"],
                        "sensitive_attribute_value": "ALL",
                        **metrics_year,
                    })
                    
                    # Compute metrics stratified by sensitive attribute values
                    stratified_metrics = _compute_stratified_metrics(
                        y_year, y_year_pred, y_year_proba, A_year, data_cfg["sensitive_attribute"]
                    )
                    for strat_entry in stratified_metrics:
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "retrain",
                            "year": test_period,
                            **strat_entry
                        })

                    primary_key = (method, data_cfg["task"], data_cfg["sensitive_attribute"])
                    retrain_metric_accumulator.setdefault(primary_key, []).append(metrics_year)

                    # Secondary attribute on yearly data
                    if "RAC1P" in X_year.columns and data_cfg["sensitive_attribute"] != "RAC1P":
                        A_year_race = (X_year["RAC1P"] == 1).astype(int)
                        metrics_year_race = compute_metrics(y_year, y_year_pred, y_year_proba, A_year_race)

                        # Save overall yearly metrics for secondary attribute (used by gap plots)
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "retrain",
                            "year": test_period,
                            "sensitive_attribute": "RAC1P",
                            "sensitive_attribute_value": "ALL",
                            **metrics_year_race,
                        })
                        
                        # Stratified metrics for secondary attribute
                        stratified_race = _compute_stratified_metrics(
                            y_year, y_year_pred, y_year_proba, A_year_race, "RAC1P"
                        )
                        for strat_entry in stratified_race:
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "retrain",
                                "year": test_period,
                                **strat_entry
                            })
                        
                        race_key = (method, data_cfg["task"], "RAC1P")
                        retrain_metric_accumulator.setdefault(race_key, []).append(metrics_year_race)

                    progress.update(f"seed={seed} method={method} year={test_period} retrain")

            # Aggregate retrain yearly metrics into overall rows used by summary/tables
            for (method, task, sensitive_attribute), metric_list in retrain_metric_accumulator.items():
                if not metric_list:
                    continue
                metric_names = metric_list[0].keys()
                aggregated_metrics = {
                    metric_name: float(np.nanmean([metrics[metric_name] for metrics in metric_list]))
                    for metric_name in metric_names
                }
                results.append(
                    {
                        "seed": seed,
                        "method": method,
                        "task": task,
                        "maintenance": "retrain",
                        "sensitive_attribute": sensitive_attribute,
                        **aggregated_metrics,
                    }
                )

    results_df = pd.DataFrame(results)
    results_path = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)

    # Summary with mean, std, and 95% CI
    # For temporal mode with maintenance column, group by method+maintenance+task+sensitive_attribute
    if mode == "temporal" and "maintenance" in results_df.columns:
        summary_ci = compute_confidence_intervals(results_df, ci=0.95, group_by=["method", "maintenance", "task", "sensitive_attribute"])
    else:
        summary_ci = compute_confidence_intervals(results_df, ci=0.95, group_by=["method", "task", "sensitive_attribute"])
    
    summary_ci_path = output_dir / "benchmark_summary_ci.csv"
    summary_ci.to_csv(summary_ci_path, index=False)
    
    # Generate attribute-specific summary tables (SEX, RAC1P, etc)
    generate_summary_tables_by_attribute(summary_ci, output_dir)
    
    # Statistical tests vs baseline
    if len(results_df["method"].unique()) > 1:
        stratify_cols = [
            col for col in ["maintenance", "task", "sensitive_attribute"] if col in results_df.columns
        ]
        sig_tests = statistical_tests_vs_baseline(results_df, stratify_by=stratify_cols)
        tests_path = output_dir / "benchmark_statistical_tests.csv"
        sig_tests.to_csv(tests_path, index=False)

    summary_group_cols = ["method"]
    if "maintenance" in results_df.columns:
        summary_group_cols.append("maintenance")
    numeric_cols = [col for col in results_df.select_dtypes(include=[np.number]).columns if col not in ["seed", "year"]]
    summary = (
        results_df.groupby(summary_group_cols)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary = flatten_summary_columns(summary)
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    
    # Generate static plots for static benchmarks
    if mode == "static":
        plot_static_comparison(summary_ci, output_dir)

    if mode == "temporal" and results_by_year:
        results_by_year_df = pd.DataFrame(results_by_year)
        results_by_year_path = output_dir / "benchmark_results_by_year.csv"
        results_by_year_df.to_csv(results_by_year_path, index=False)

        # Include maintenance and sensitive_attribute_value in groupby if they exist
        groupby_cols = ["year", "method"]
        if "maintenance" in results_by_year_df.columns:
            groupby_cols.append("maintenance")
        if "sensitive_attribute_value" in results_by_year_df.columns:
            groupby_cols.append("sensitive_attribute_value")
        
        numeric_year_cols = [
            col for col in results_by_year_df.select_dtypes(include=[np.number]).columns if col not in ["seed", "year"]
        ]
        summary_by_year = (
            results_by_year_df.groupby(groupby_cols, sort=True)[numeric_year_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary_by_year = flatten_summary_columns(summary_by_year)
        summary_by_year_path = output_dir / "benchmark_summary_by_year.csv"
        summary_by_year.to_csv(summary_by_year_path, index=False)

        plot_temporal_metrics(results_by_year_df, output_dir)
        plot_temporal_comparison_by_year(results_by_year_df, output_dir)
        plot_original_vs_updated(results_by_year_df, output_dir)
        plot_original_vs_updated_by_attribute(results_by_year_df, output_dir)

    elapsed_seconds = time.perf_counter() - start_time
    meta = {
        "run_id": run_id,
        "config_path": str(config_path),
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "experiment": exp_cfg["name"],
        "methods": methods,
        "seeds": seeds,
        "results": str(results_path),
        "summary": str(summary_path),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    if compare_outputs:
        comparison_tables = []
        compare_paths = [summary_path] + [Path(p) for p in compare_outputs]
        for path in compare_paths:
            df = pd.read_csv(path)
            df["source"] = path.stem
            comparison_tables.append(df)
        comparison_df = pd.concat(comparison_tables, ignore_index=True)
        comparison_path = output_dir / "benchmark_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        meta["comparison"] = str(comparison_path)

    meta_path = output_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    history_path = _append_run_history(
        base_output_dir,
        {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "experiment": exp_cfg["name"],
            "task": data_cfg.get("task", "unknown"),
            "mode": mode,
            "output_dir": str(output_dir),
            "results": str(results_path),
            "summary": str(summary_path),
            "elapsed_seconds": round(elapsed_seconds, 2),
        },
    )

    progress.close()

    # Generate initial performance table for paper
    initial_table_path = None
    if mode == "temporal" and results_by_year:
        initial_period = min(results_by_year_df["year"])
        initial_subset = results_by_year_df[results_by_year_df["year"] == initial_period].copy()
        if "maintenance" in initial_subset.columns:
            initial_subset = initial_subset[initial_subset["maintenance"] == "no-retrain"]
        if "task" in initial_subset.columns:
            initial_subset = initial_subset[initial_subset["task"] == data_cfg.get("task")]
        if "sensitive_attribute" in initial_subset.columns:
            initial_subset = initial_subset[initial_subset["sensitive_attribute"] == data_cfg.get("sensitive_attribute")]

        if not initial_subset.empty:
            initial_summary = compute_confidence_intervals(initial_subset, ci=0.95, group_by=["method"])
            initial_table_path = _generate_initial_performance_table(
                initial_summary,
                data_cfg.get("task", "income"),
                output_dir,
            )

    if initial_table_path is None:
        initial_table_path = _generate_initial_performance_table(
            summary_ci,
            data_cfg.get("task", "income"),
            output_dir,
            maintenance="no-retrain" if mode == "temporal" else None,
            sensitive_attribute=data_cfg.get("sensitive_attribute") if mode == "temporal" else None,
        )

    elapsed_minutes = elapsed_seconds / 60
    print("✓ Benchmark complete")
    print(f"  Base output dir: {base_output_dir}")
    print(f"  Run output dir: {output_dir}")
    print(f"  Results: {results_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Summary with CI: {summary_ci_path}")
    if len(results_df["method"].unique()) > 1:
        print(f"  Statistical tests: {tests_path}")
    if initial_table_path is not None:
        print(f"  Initial performance table: {initial_table_path}")
    print(f"  Run history: {history_path}")
    print(f"  Elapsed time: {elapsed_minutes:.1f} min ({elapsed_seconds:.1f} s)")
