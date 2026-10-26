"""Recompute the headline numbers and Table III of the paper from the versioned runs.

Convention (paper, Section III-C):
  * a calendar-year value is the mean over that year's four within-year folds
    (fold years are encoded YYYY.00/.25/.50/.75, so calendar year = floor);
  * 2014 values are the development hold-out (phase != "temporal");
  * each sensitive attribute is read from the run configured for that attribute;
  * uncertainty is a 95% t-interval over the 20 seed-level estimates.

Usage:
    python reproduce/headline_numbers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/folktables"
RUNS = {
    ("income", "SEX"): R / "benchmark_temporal/folktables_benchmark_temporal/20260507_165711",
    ("income", "RAC1P"): R / "benchmark_temporal_race/folktables_benchmark_temporal_race/20260508_111012",
    ("employment", "SEX"): R / "benchmark_temporal_employment/folktables_benchmark_temporal_employment/20260507_232826",
    ("employment", "RAC1P"): R / "benchmark_temporal_employment_race/folktables_benchmark_temporal_employment_race/20260508_173829",
}
POLICIES = {
    "no-retrain": "Frozen",
    "threshold-recalibration": "Thresh. recal.",
    "last-year-retrain": "Last-year",
    "cumulative-retrain": "Cumulative",
    "cumulative-retrain-fixed-n": "Cumul. fixed-n",
}


def seed_year(task, attr, method="baseline", group="ALL"):
    """Seed-level calendar-year values (mean of the four folds); 2014 = development hold-out."""
    d = pd.read_csv(RUNS[(task, attr)] / "benchmark_results_by_year.csv")
    d = d[(d.sensitive_attribute == attr) & (d.method == method)
          & (d.sensitive_attribute_value.astype(str) == group)]
    d = d.assign(yr=np.where(d.phase == "temporal", np.floor(d.year), 2014).astype(int))
    return d.groupby(["maintenance", "phase", "yr", "seed"], as_index=False).mean(numeric_only=True)


def ci95(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def frozen_2014(s):
    return s[s.phase != "temporal"]


def at_2018(s, policy):
    return s[(s.phase == "temporal") & (s.yr == 2018) & (s.maintenance == policy)]


def main():
    # Fairness drift: income x race, frozen baseline, equalized-odds gap 2014 -> 2018
    s = seed_year("income", "RAC1P")
    m0, l0, h0 = ci95(frozen_2014(s).eo_gap)
    m1, l1, h1 = ci95(at_2018(s, "no-retrain").eo_gap)
    print("Fairness drift (income x race, frozen baseline, EO gap)")
    print(f"  2014: {m0:.4f}  95% CI [{l0:.4f}, {h0:.4f}]")
    print(f"  2018: {m1:.4f}  95% CI [{l1:.4f}, {h1:.4f}]   change {100 * (m1 - m0) / m0:+.1f}%")
    rec = at_2018(s, "threshold-recalibration").eo_gap.mean()
    print(f"  2018 after threshold recalibration: {rec:.4f}\n")

    # Maintenance trade-off: income x sex, 2018, frozen vs threshold recalibration
    s = seed_year("income", "SEX")
    fro, rec = at_2018(s, "no-retrain"), at_2018(s, "threshold-recalibration")
    print("Maintenance trade-off (income x sex, 2018, frozen -> threshold recalibration)")
    for col, name in [("dp_gap", "DP gap"), ("eo_gap", "EO gap"), ("accuracy", "Accuracy"),
                      ("auc", "AUC"), ("positive_rate", "PPR")]:
        a, b = fro[col].mean(), rec[col].mean()
        extra = f"   change {100 * (b - a) / a:+.1f}%" if col.endswith("gap") else ""
        print(f"  {name:<9} {a:.4f} -> {b:.4f}{extra}")

    # Table III: utility of the baseline pipeline (SEX-configured runs)
    print("\nTable III (baseline pipeline, means over 20 seeds)")
    print(f"  {'Task':<11}{'Condition':<24}{'Acc':>7}{'F1':>7}{'AUC':>7}{'Brier':>7}")
    for task in ("income", "employment"):
        s = seed_year(task, "SEX")
        rows = [("2014 dev.", frozen_2014(s))] + [(f"{lab} (2018)", at_2018(s, p)) for p, lab in POLICIES.items()]
        for lab, d in rows:
            vals = "".join(f"{d[c].mean():7.3f}" for c in ("accuracy", "f1_score", "auc", "brier_score"))
            print(f"  {task.capitalize():<11}{lab:<24}{vals}")
            task = ""


if __name__ == "__main__":
    main()
