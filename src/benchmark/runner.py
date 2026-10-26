"""Benchmark runner implementation (static and temporal)."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
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
    choose_thresholds_demographic_parity,
    train_with_fairlearn_threshold_optimizer,
    train_with_fairlearn_reductions,
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
    plot_fairness_by_method,
    compute_confidence_intervals,
    statistical_tests_vs_baseline,
)


DEFAULT_EXCLUDED_FEATURE_COLUMNS = ("SEX", "RAC1P")
MAINTENANCE_ALIASES = {
    "retrain": "cumulative-retrain",
    "annual-retrain": "cumulative-retrain",
    "annual-cumulative-retrain": "cumulative-retrain",
}
RETRAIN_MAINTENANCE_STRATEGIES = {
    "cumulative-retrain",
    "last-year-retrain",
    "cumulative-retrain-fixed-n",
}


def _extract_binary_attribute(X_df: pd.DataFrame, attribute: str):
    """Extract the binary encoding used for fairness metrics."""
    if attribute == "SEX" and "SEX" in X_df.columns:
        return (X_df["SEX"] == 1).astype(int)
    if attribute == "RAC1P" and "RAC1P" in X_df.columns:
        return (X_df["RAC1P"] == 1).astype(int)
    return None


def _secondary_attribute_vectors(X_df: pd.DataFrame, primary_attribute: str) -> dict:
    """Return non-primary sensitive attributes available in a feature frame."""
    vectors = {}
    for attribute in DEFAULT_EXCLUDED_FEATURE_COLUMNS:
        if attribute == primary_attribute:
            continue
        vector = _extract_binary_attribute(X_df, attribute)
        if vector is not None:
            vectors[attribute] = vector
    return vectors


def _model_feature_columns(columns, data_cfg: dict) -> list:
    """Return feature columns used by the predictive model."""
    if data_cfg.get("allow_sensitive_features", False):
        return list(columns)

    excluded = set(data_cfg.get("excluded_feature_columns", DEFAULT_EXCLUDED_FEATURE_COLUMNS))
    model_columns = [col for col in columns if col not in excluded]
    if not model_columns:
        raise ValueError("No model feature columns remain after excluding sensitive attributes.")
    return model_columns


def _normalize_maintenance_strategy(strategy: str) -> str:
    return MAINTENANCE_ALIASES.get(strategy, strategy)


def _create_strata(y, A):
    return (np.asarray(y).astype(int) * 2 + np.asarray(A).astype(int)).astype(int)


def _split_train_val_df(X_df: pd.DataFrame, y, A, seed: int, train_size: float = 0.8):
    """Split a labeled frame into train/validation partitions stratified by label and group."""
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx,
        train_size=train_size,
        random_state=seed,
        stratify=_create_strata(y, A),
    )
    return (
        X_df.iloc[train_idx].reset_index(drop=True),
        X_df.iloc[val_idx].reset_index(drop=True),
        np.asarray(y)[train_idx].astype(int),
        np.asarray(y)[val_idx].astype(int),
        np.asarray(A)[train_idx].astype(int),
        np.asarray(A)[val_idx].astype(int),
    )


def _sample_fixed_n(X_df: pd.DataFrame, y, A, n: int, seed: int):
    """Subsample a training frame to a fixed size for controlled retraining ablations."""
    if n is None or len(y) <= n:
        return X_df, y, A
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return (
        X_df.iloc[idx].reset_index(drop=True),
        np.asarray(y)[idx].astype(int),
        np.asarray(A)[idx].astype(int),
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
    utility_config = config.get("threshold_selection", {})

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
        post_cfg = config.get("postprocessing", {})
        if post_cfg.get("backend", "fairlearn_threshold_optimizer") == "grid_search":
            model = make_model(seed)
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            thresholds = choose_thresholds_equalized_odds(
                y_val, y_val_proba, A_val, grid=threshold_grid, utility_config=utility_config
            )
            return model, thresholds

        model = train_with_fairlearn_threshold_optimizer(
            X_train,
            y_train,
            X_val,
            y_val,
            A_val,
            seed,
            constraint="equalized_odds",
            objective=post_cfg.get("objective", "balanced_accuracy_score"),
            grid_size=int(post_cfg.get("grid_size", 100)),
        )
        return model, None
    
    elif method == "demographic_parity":
        post_cfg = config.get("postprocessing", {})
        if post_cfg.get("backend", "fairlearn_threshold_optimizer") == "grid_search":
            model = make_model(seed)
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            thresholds = choose_thresholds_demographic_parity(
                y_val, y_val_proba, A_val, grid=threshold_grid, utility_config=utility_config
            )
            return model, thresholds

        model = train_with_fairlearn_threshold_optimizer(
            X_train,
            y_train,
            X_val,
            y_val,
            A_val,
            seed,
            constraint="demographic_parity",
            objective=post_cfg.get("objective", "balanced_accuracy_score"),
            grid_size=int(post_cfg.get("grid_size", 100)),
        )
        return model, None
    
    elif method == "fairness_constraint":
        constraint_cfg = config.get("fairness_constraint", {})
        backend = constraint_cfg.get("backend", "fairlearn_reductions")
        if backend == "legacy_surrogate":
            num_iters = int(constraint_cfg.get("num_iters", 8))
            lr = float(constraint_cfg.get("lr", 0.1))
            model = train_with_lagrangian(X_train, y_train, A_train, seed, num_iters, lr)
        else:
            model = train_with_fairlearn_reductions(
                X_train,
                y_train,
                A_train,
                seed,
                constraint=constraint_cfg.get("constraint", "demographic_parity"),
                eps=float(constraint_cfg.get("eps", 0.02)),
                max_iter=int(constraint_cfg.get("max_iter", 50)),
            )
        return model, None
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _predict_with_method(method, model, thresholds, X_test, A_test):
    """Get predictions from a trained model."""
    predictor = thresholds if hasattr(thresholds, "predict_with_sensitive") else model
    if hasattr(predictor, "predict_proba_with_sensitive"):
        y_proba = predictor.predict_proba_with_sensitive(X_test, A_test)[:, 1]
    else:
        y_proba = predictor.predict_proba(X_test)[:, 1]

    if hasattr(predictor, "predict_with_sensitive"):
        y_pred = predictor.predict_with_sensitive(X_test, A_test).astype(int)
        return y_pred, y_proba
    
    if thresholds is not None and "global" in thresholds:
        y_pred = (y_proba >= thresholds["global"]).astype(int)
    elif method in {"equalized_odds", "demographic_parity"} and thresholds is not None:
        y_pred = np.where(
            A_test == 0,
            y_proba >= thresholds[0],
            y_proba >= thresholds[1],
        ).astype(int)
    else:
        y_pred = (y_proba >= 0.5).astype(int)
    
    return y_pred, y_proba


def _choose_global_threshold(y_true, y_proba, grid):
    """Choose a single threshold maximizing validation F1, breaking ties by accuracy."""
    best_threshold = 0.5
    best_f1 = -np.inf
    best_acc = -np.inf
    y_true = np.asarray(y_true).astype(int)

    for threshold in grid:
        y_pred = (y_proba >= threshold).astype(int)
        tp = float(((y_true == 1) & (y_pred == 1)).sum())
        fp = float(((y_true == 0) & (y_pred == 1)).sum())
        fn = float(((y_true == 1) & (y_pred == 0)).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        acc = float(np.mean(y_pred == y_true))
        if f1 > best_f1 or (np.isclose(f1, best_f1) and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_threshold = float(threshold)

    return {"global": best_threshold}


def _calibrate_thresholds_for_method(method, model, X_cal, y_cal, A_cal, threshold_grid, config):
    """Select thresholds for a fixed trained scorer on calibration data."""
    if hasattr(model, "recalibrate"):
        return model.recalibrate(X_cal, y_cal, A_cal)

    y_cal_proba = model.predict_proba(X_cal)[:, 1]
    utility_config = config.get("threshold_selection", {})
    if method == "equalized_odds":
        return choose_thresholds_equalized_odds(
            y_cal, y_cal_proba, A_cal, grid=threshold_grid, utility_config=utility_config
        )
    if method == "demographic_parity":
        return choose_thresholds_demographic_parity(
            y_cal, y_cal_proba, A_cal, grid=threshold_grid, utility_config=utility_config
        )
    return _choose_global_threshold(y_cal, y_cal_proba, threshold_grid)


def _threshold_metadata(thresholds):
    """Serialize group-specific thresholds into flat result columns."""
    if thresholds is None:
        return {"threshold_group0": np.nan, "threshold_group1": np.nan, "threshold_global": np.nan}
    if hasattr(thresholds, "predict_with_sensitive"):
        return {"threshold_group0": np.nan, "threshold_group1": np.nan, "threshold_global": np.nan}
    return {
        "threshold_group0": float(thresholds.get(0, np.nan)),
        "threshold_group1": float(thresholds.get(1, np.nan)),
        "threshold_global": float(thresholds.get("global", np.nan)),
    }


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
    
    # Model-maintenance policies. Legacy "retrain" is treated as cumulative retraining.
    maintenance_strategies = [
        _normalize_maintenance_strategy(strategy)
        for strategy in config.get("maintenance_strategies", ["no-retrain"])
    ]

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
            split_ratio = [0.7, 0.1, 0.2]  # train, val, test
            # Seed-specific splits are created inside the seed loop so CIs
            # capture development-split variability as well as model randomness.
            X_train_df = X_val_df = X_test_dev_df = None
            y_train = y_val = y_test_dev = None
            A_train = A_val = A_test_dev = None
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
            keep_sensitive_cols=True,
        )

        # If no temporal years are loaded, infer base columns from development data
        if base_columns is None:
            base_columns = X_train_df.columns

        if X_val_df is not None:
            X_val_df = X_val_df.reindex(columns=base_columns, fill_value=0)
        if X_train_df is not None:
            X_train_df = X_train_df.reindex(columns=base_columns, fill_value=0)

        if X_test_dev_df is not None:
            X_test_dev_df = X_test_dev_df.reindex(columns=base_columns, fill_value=0)

        X_test_df = X_test_df.reindex(columns=base_columns, fill_value=0)
        model_columns = _model_feature_columns(base_columns, data_cfg)

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
        for strategy in maintenance_strategies:
            if strategy in RETRAIN_MAINTENANCE_STRATEGIES or strategy == "threshold-recalibration":
                progress_total += num_seeds * num_methods * len(retrain_test_periods)
    else:
        # Static mode: keep_sensitive_cols defaults to False so the primary
        # sensitive attribute is excluded from the feature matrix.  A_all still
        # captures the binary sensitive-attribute vector for fairness metrics.
        X_df, y_all, A_all = load_folktables(
            task=data_cfg["task"],
            states=data_cfg["states"],
            years=data_cfg["years"],
            sensitive_attribute=data_cfg["sensitive_attribute"],
            max_samples=max_samples,
            random_state=sample_seed,
        )
        progress_total = ProgressCalculator.calculate_static_steps(
            num_seeds=len(seeds),
            num_methods=len(methods),
        )

    progress = ProgressTracker(progress_total)

    results = []
    for seed in seeds:
        if mode == "temporal":
            if same_year_split:
                X_train_raw, X_val_raw, X_test_dev_raw, y_train, y_val, y_test_dev, A_train, A_val, A_test_dev = stratified_split(
                    X_pool_df.values, y_pool, A_pool, seed=seed, split=split_ratio
                )
                X_train_df = pd.DataFrame(X_train_raw, columns=X_pool_df.columns).reindex(columns=base_columns, fill_value=0)
                X_val_df = pd.DataFrame(X_val_raw, columns=X_pool_df.columns).reindex(columns=base_columns, fill_value=0)
                X_test_dev_df = pd.DataFrame(X_test_dev_raw, columns=X_pool_df.columns).reindex(columns=base_columns, fill_value=0)

            X_train = X_train_df[model_columns].values
            X_val = X_val_df[model_columns].values
            X_test = X_test_df[model_columns].values
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
        baseline_train_n = len(y_train)
        base_model_cache = {}

        # === NO-RETRAIN SCENARIO ===
        if "no-retrain" in maintenance_strategies:
            for method in methods:
                model, thresholds = _train_method_on_data(
                    method, X_train, y_train, A_train, X_val, y_val, A_val, seed, threshold_grid, config
                )
                threshold_meta = _threshold_metadata(thresholds)
                base_model_cache[method] = (model, thresholds)
                
                # Test on aggregated test set
                y_pred, y_proba = _predict_with_method(method, model, thresholds, X_test, A_test)
                
                # Compute metrics on primary attribute (SEX in config)
                metrics = compute_metrics(y_test, y_pred, y_proba, A_test)
                result_entry = {
                    "seed": seed,
                    "method": method,
                    "task": data_cfg["task"],
                    "sensitive_attribute": data_cfg["sensitive_attribute"],
                    **threshold_meta,
                    **metrics
                }
                if mode != "temporal":
                    results.append(result_entry)

                temporal_primary_metrics = []
                temporal_secondary_metrics = {}
                
                progress.update(f"seed={seed} method={method} test=all")

                # Evaluate the 2014 hold-out split explicitly so baseline tables/plots
                # refer to the actual development-year test set rather than the first
                # temporal fold from 2015.
                if mode == "temporal" and X_test_dev_df is not None and y_test_dev is not None and A_test_dev is not None:
                    X_dev_scaled = scaler.transform(X_test_dev_df[model_columns].values)
                    y_dev_pred, y_dev_proba = _predict_with_method(
                        method, model, thresholds, X_dev_scaled, A_test_dev
                    )
                    dev_year = float(max(int(y) for y in train_years))
                    dev_metrics = compute_metrics(y_test_dev, y_dev_pred, y_dev_proba, A_test_dev)

                    results_by_year.append({
                        "seed": seed,
                        "method": method,
                        "task": data_cfg["task"],
                        "maintenance": "no-retrain",
                        "phase": "baseline",
                        "year": dev_year,
                        "sensitive_attribute": data_cfg["sensitive_attribute"],
                        "sensitive_attribute_value": "ALL",
                        **threshold_meta,
                        **dev_metrics,
                    })

                    baseline_stratified = _compute_stratified_metrics(
                        y_test_dev, y_dev_pred, y_dev_proba, A_test_dev, data_cfg["sensitive_attribute"]
                    )
                    for strat_entry in baseline_stratified:
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "no-retrain",
                            "phase": "baseline",
                            "year": dev_year,
                            **threshold_meta,
                            **strat_entry,
                        })

                    for secondary_attr, A_dev_secondary in _secondary_attribute_vectors(
                        X_test_dev_df, data_cfg["sensitive_attribute"]
                    ).items():
                        dev_metrics_secondary = compute_metrics(
                            y_test_dev, y_dev_pred, y_dev_proba, A_dev_secondary
                        )
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "no-retrain",
                            "phase": "baseline",
                            "year": dev_year,
                            "sensitive_attribute": secondary_attr,
                            "sensitive_attribute_value": "ALL",
                            **threshold_meta,
                            **dev_metrics_secondary,
                        })

                        baseline_stratified_secondary = _compute_stratified_metrics(
                            y_test_dev, y_dev_pred, y_dev_proba, A_dev_secondary, secondary_attr
                        )
                        for strat_entry in baseline_stratified_secondary:
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "phase": "baseline",
                                "year": dev_year,
                                **threshold_meta,
                                **strat_entry,
                            })

                # Test on each year
                if mode == "temporal":
                    for year, (X_year, y_year, A_year) in test_by_year.items():
                        X_year = X_year.reindex(columns=base_columns, fill_value=0)
                        X_year_scaled = scaler.transform(X_year[model_columns].values)
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
                            "phase": "temporal",
                            "year": year,
                            "sensitive_attribute": data_cfg["sensitive_attribute"],
                            "sensitive_attribute_value": "ALL",
                            **threshold_meta,
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
                                "phase": "temporal",
                                "year": year,
                                **threshold_meta,
                                **strat_entry
                            })

                        if year in summary_test_periods_set:
                            temporal_primary_metrics.append(metrics_year)
                        
                        # Secondary attributes on yearly data
                        for secondary_attr, A_year_secondary in _secondary_attribute_vectors(
                            X_year, data_cfg["sensitive_attribute"]
                        ).items():
                            metrics_year_secondary = compute_metrics(
                                y_year, y_year_pred, y_year_proba, A_year_secondary
                            )

                            # Save overall yearly metrics for secondary attribute (used by gap plots)
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "phase": "temporal",
                                "year": year,
                                "sensitive_attribute": secondary_attr,
                                "sensitive_attribute_value": "ALL",
                                **threshold_meta,
                                **metrics_year_secondary,
                            })
                            
                            # Stratified metrics for secondary attribute
                            stratified_secondary = _compute_stratified_metrics(
                                y_year, y_year_pred, y_year_proba, A_year_secondary, secondary_attr
                            )
                            for strat_entry in stratified_secondary:
                                results_by_year.append({
                                    "seed": seed,
                                    "method": method,
                                    "task": data_cfg["task"],
                                    "maintenance": "no-retrain",
                                    "phase": "temporal",
                                    "year": year,
                                    **threshold_meta,
                                    **strat_entry
                                })

                            if year in summary_test_periods_set:
                                temporal_secondary_metrics.setdefault(secondary_attr, []).append(metrics_year_secondary)
                        
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
                                **threshold_meta,
                                **aggregated_primary_metrics,
                            }
                        )

                    for secondary_attr, metric_list in temporal_secondary_metrics.items():
                        metric_names_secondary = metric_list[0].keys()
                        aggregated_secondary_metrics = {
                            metric_name: float(np.nanmean([metrics_row[metric_name] for metrics_row in metric_list]))
                            for metric_name in metric_names_secondary
                        }
                        results.append(
                            {
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "no-retrain",
                                "sensitive_attribute": secondary_attr,
                                **threshold_meta,
                                **aggregated_secondary_metrics,
                            }
                        )

        # === THRESHOLD-RECALIBRATION SCENARIO ===
        if "threshold-recalibration" in maintenance_strategies and mode == "temporal":
            threshold_metric_accumulator = {}
            for test_period in retrain_test_periods:
                eval_year = int(np.floor(test_period))
                calibration_year = eval_year - 1

                if calibration_year in {int(y) for y in train_years}:
                    X_cal_df = X_val_df.copy()
                    y_cal = y_val
                    A_cal = A_val
                else:
                    X_cal_df, y_cal, A_cal = load_folktables(
                        task=data_cfg["task"],
                        states=data_cfg["states"],
                        years=[calibration_year],
                        sensitive_attribute=data_cfg["sensitive_attribute"],
                        max_samples=max_samples,
                        random_state=int(sample_seed + 7 + eval_year),
                        keep_sensitive_cols=True,
                    )
                    X_cal_df = X_cal_df.reindex(columns=base_columns, fill_value=0)

                X_cal = scaler.transform(X_cal_df[model_columns].values)
                X_year, y_year, A_year = test_by_year[test_period]
                X_year = X_year.reindex(columns=base_columns, fill_value=0)
                X_year_scaled = scaler.transform(X_year[model_columns].values)

                for method in methods:
                    if method in base_model_cache:
                        model, _ = base_model_cache[method]
                    else:
                        model, thresholds = _train_method_on_data(
                            method, X_train, y_train, A_train, X_val, y_val, A_val, seed, threshold_grid, config
                        )
                        base_model_cache[method] = (model, thresholds)

                    thresholds = _calibrate_thresholds_for_method(
                        method, model, X_cal, y_cal, A_cal, threshold_grid, config
                    )
                    threshold_meta = _threshold_metadata(thresholds)
                    y_year_pred, y_year_proba = _predict_with_method(
                        method, model, thresholds, X_year_scaled, A_year
                    )
                    metrics_year = compute_metrics(y_year, y_year_pred, y_year_proba, A_year)
                    primary_key = (method, data_cfg["task"], data_cfg["sensitive_attribute"])
                    threshold_metric_accumulator.setdefault(primary_key, []).append(metrics_year)

                    results_by_year.append({
                        "seed": seed,
                        "method": method,
                        "task": data_cfg["task"],
                        "maintenance": "threshold-recalibration",
                        "phase": "temporal",
                        "year": test_period,
                        "sensitive_attribute": data_cfg["sensitive_attribute"],
                        "sensitive_attribute_value": "ALL",
                        **threshold_meta,
                        **metrics_year,
                    })

                    stratified_metrics = _compute_stratified_metrics(
                        y_year, y_year_pred, y_year_proba, A_year, data_cfg["sensitive_attribute"]
                    )
                    for strat_entry in stratified_metrics:
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "threshold-recalibration",
                            "phase": "temporal",
                            "year": test_period,
                            **threshold_meta,
                            **strat_entry,
                        })

                    for secondary_attr, A_year_secondary in _secondary_attribute_vectors(
                        X_year, data_cfg["sensitive_attribute"]
                    ).items():
                        metrics_year_secondary = compute_metrics(
                            y_year, y_year_pred, y_year_proba, A_year_secondary
                        )
                        secondary_key = (method, data_cfg["task"], secondary_attr)
                        threshold_metric_accumulator.setdefault(secondary_key, []).append(metrics_year_secondary)
                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": "threshold-recalibration",
                            "phase": "temporal",
                            "year": test_period,
                            "sensitive_attribute": secondary_attr,
                            "sensitive_attribute_value": "ALL",
                            **threshold_meta,
                            **metrics_year_secondary,
                        })

                        stratified_secondary = _compute_stratified_metrics(
                            y_year, y_year_pred, y_year_proba, A_year_secondary, secondary_attr
                        )
                        for strat_entry in stratified_secondary:
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": "threshold-recalibration",
                                "phase": "temporal",
                                "year": test_period,
                                **threshold_meta,
                                **strat_entry,
                            })

                    progress.update(f"seed={seed} method={method} year={test_period} threshold-recalibration")

            for (method, task, sensitive_attribute), metric_list in threshold_metric_accumulator.items():
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
                        "maintenance": "threshold-recalibration",
                        "sensitive_attribute": sensitive_attribute,
                        **aggregated_metrics,
                    }
                )

        # === RETRAIN SCENARIO ===
        if any(strategy in RETRAIN_MAINTENANCE_STRATEGIES for strategy in maintenance_strategies) and mode == "temporal":
            sorted_test_periods = retrain_test_periods
            temporal_eval_years = sorted({int(np.floor(p)) for p in sorted_test_periods})
            active_retrain_strategies = [
                strategy for strategy in maintenance_strategies if strategy in RETRAIN_MAINTENANCE_STRATEGIES
            ]

            for maintenance in active_retrain_strategies:
                retrain_metric_accumulator = {}

                for test_period in sorted_test_periods:
                    eval_year = int(np.floor(test_period))
                    base_train_years = sorted({int(y) for y in train_years})

                    if maintenance == "last-year-retrain":
                        previous_year_candidates = base_train_years + [
                            year for year in temporal_eval_years if year < eval_year
                        ]
                        previous_year = max(previous_year_candidates)
                        if same_year_split and previous_year in set(base_train_years):
                            X_retrain_df = X_train_df.copy()
                            y_retrain = y_train
                            A_retrain = A_train
                            X_val_retrain_df = X_val_df.copy()
                            y_val_retrain = y_val
                            A_val_retrain = A_val
                        else:
                            X_previous_df, y_previous, A_previous = load_folktables(
                                task=data_cfg["task"],
                                states=data_cfg["states"],
                                years=[previous_year],
                                sensitive_attribute=data_cfg["sensitive_attribute"],
                                max_samples=max_samples,
                                random_state=int(sample_seed + 8 + eval_year),
                                keep_sensitive_cols=True,
                            )
                            X_previous_df = X_previous_df.reindex(columns=base_columns, fill_value=0)
                            (
                                X_retrain_df,
                                X_val_retrain_df,
                                y_retrain,
                                y_val_retrain,
                                A_retrain,
                                A_val_retrain,
                            ) = _split_train_val_df(
                                X_previous_df,
                                y_previous,
                                A_previous,
                                seed=int(seed + 1000 + eval_year),
                                train_size=0.8,
                            )
                    else:
                        # Cumulative training history: base development years + prior temporal years.
                        prior_temporal_years = [year for year in temporal_eval_years if year < eval_year]
                        train_years_for_retrain = sorted(set(base_train_years + prior_temporal_years))
                        if not train_years_for_retrain:
                            train_years_for_retrain = base_train_years

                        if same_year_split:
                            retrain_frames = [X_train_df.copy()]
                            y_parts = [y_train]
                            A_parts = [A_train]

                            prior_years_for_retrain = [
                                year for year in train_years_for_retrain if year not in set(base_train_years)
                            ]
                            if prior_years_for_retrain:
                                X_prior_df, y_prior, A_prior = load_folktables(
                                    task=data_cfg["task"],
                                    states=data_cfg["states"],
                                    years=prior_years_for_retrain,
                                    sensitive_attribute=data_cfg["sensitive_attribute"],
                                    max_samples=max_samples,
                                    random_state=int(sample_seed + 5 + eval_year),
                                    keep_sensitive_cols=True,
                                )
                                retrain_frames.append(X_prior_df.reindex(columns=base_columns, fill_value=0))
                                y_parts.append(y_prior)
                                A_parts.append(A_prior)

                            X_retrain_df = pd.concat(retrain_frames, axis=0).reset_index(drop=True)
                            y_retrain = np.concatenate(y_parts).astype(int)
                            A_retrain = np.concatenate(A_parts).astype(int)
                        else:
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

                        if maintenance == "cumulative-retrain-fixed-n":
                            X_retrain_df, y_retrain, A_retrain = _sample_fixed_n(
                                X_retrain_df,
                                y_retrain,
                                A_retrain,
                                n=baseline_train_n,
                                seed=int(seed + 2000 + eval_year),
                            )

                        # Keep validation disjoint from retraining data. Reusing the original
                        # development validation split avoids overlap with cumulative training data.
                        X_val_retrain_df = X_val_df.copy()
                        y_val_retrain = y_val
                        A_val_retrain = A_val

                    scaler_retrain = StandardScaler()
                    X_retrain = scaler_retrain.fit_transform(X_retrain_df[model_columns].values)
                    X_val_retrain = scaler_retrain.transform(X_val_retrain_df[model_columns].values)

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
                        threshold_meta = _threshold_metadata(thresholds)

                        X_year, y_year, A_year = test_by_year[test_period]
                        X_year = X_year.reindex(columns=base_columns, fill_value=0)
                        X_year_scaled = scaler_retrain.transform(X_year[model_columns].values)
                        y_year_pred, y_year_proba = _predict_with_method(
                            method, model, thresholds, X_year_scaled, A_year
                        )
                        metrics_year = compute_metrics(y_year, y_year_pred, y_year_proba, A_year)

                        results_by_year.append({
                            "seed": seed,
                            "method": method,
                            "task": data_cfg["task"],
                            "maintenance": maintenance,
                            "phase": "temporal",
                            "year": test_period,
                            "sensitive_attribute": data_cfg["sensitive_attribute"],
                            "sensitive_attribute_value": "ALL",
                            **threshold_meta,
                            **metrics_year,
                        })

                        stratified_metrics = _compute_stratified_metrics(
                            y_year, y_year_pred, y_year_proba, A_year, data_cfg["sensitive_attribute"]
                        )
                        for strat_entry in stratified_metrics:
                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": maintenance,
                                "phase": "temporal",
                                "year": test_period,
                                **threshold_meta,
                                **strat_entry,
                            })

                        primary_key = (method, data_cfg["task"], data_cfg["sensitive_attribute"])
                        retrain_metric_accumulator.setdefault(primary_key, []).append(metrics_year)

                        for secondary_attr, A_year_secondary in _secondary_attribute_vectors(
                            X_year, data_cfg["sensitive_attribute"]
                        ).items():
                            metrics_year_secondary = compute_metrics(
                                y_year, y_year_pred, y_year_proba, A_year_secondary
                            )

                            results_by_year.append({
                                "seed": seed,
                                "method": method,
                                "task": data_cfg["task"],
                                "maintenance": maintenance,
                                "phase": "temporal",
                                "year": test_period,
                                "sensitive_attribute": secondary_attr,
                                "sensitive_attribute_value": "ALL",
                                **threshold_meta,
                                **metrics_year_secondary,
                            })

                            stratified_secondary = _compute_stratified_metrics(
                                y_year, y_year_pred, y_year_proba, A_year_secondary, secondary_attr
                            )
                            for strat_entry in stratified_secondary:
                                results_by_year.append({
                                    "seed": seed,
                                    "method": method,
                                    "task": data_cfg["task"],
                                    "maintenance": maintenance,
                                    "phase": "temporal",
                                    "year": test_period,
                                    **threshold_meta,
                                    **strat_entry,
                                })

                            secondary_key = (method, data_cfg["task"], secondary_attr)
                            retrain_metric_accumulator.setdefault(secondary_key, []).append(metrics_year_secondary)

                        progress.update(f"seed={seed} method={method} year={test_period} {maintenance}")

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
                            "maintenance": maintenance,
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
    if "task" in results_df.columns:
        summary_group_cols.append("task")
    if "sensitive_attribute" in results_df.columns:
        summary_group_cols.append("sensitive_attribute")
    numeric_cols = [col for col in results_df.select_dtypes(include=[np.number]).columns if col not in ["seed", "year"]]
    summary = (
        results_df.groupby(summary_group_cols)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary = flatten_summary_columns(summary)
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    
    if mode == "temporal" and results_by_year:
        results_by_year_df = pd.DataFrame(results_by_year)
        results_by_year_path = output_dir / "benchmark_results_by_year.csv"
        results_by_year_df.to_csv(results_by_year_path, index=False)

        # Include maintenance and sensitive_attribute_value in groupby if they exist
        groupby_cols = ["year", "method"]
        if "phase" in results_by_year_df.columns:
            groupby_cols.append("phase")
        if "maintenance" in results_by_year_df.columns:
            groupby_cols.append("maintenance")
        if "task" in results_by_year_df.columns:
            groupby_cols.append("task")
        if "sensitive_attribute" in results_by_year_df.columns:
            groupby_cols.append("sensitive_attribute")
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

        plot_original_vs_updated(results_by_year_df, output_dir)
        plot_original_vs_updated_by_attribute(results_by_year_df, output_dir)
        plot_fairness_by_method(results_by_year_df, output_dir)

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
    if mode == "temporal":
        meta["model_feature_columns"] = list(model_columns)
        meta["excluded_feature_columns"] = [
            col for col in DEFAULT_EXCLUDED_FEATURE_COLUMNS if col in list(base_columns)
        ]

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
        initial_subset = results_by_year_df.copy()
        if "phase" in initial_subset.columns:
            initial_subset = initial_subset[initial_subset["phase"] == "baseline"]
        else:
            initial_period = min(initial_subset["year"])
            initial_subset = initial_subset[initial_subset["year"] == initial_period]
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

    # Return run metadata for external use
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "base_output_dir": str(base_output_dir),
        "task": data_cfg.get("task", "unknown"),
    }
