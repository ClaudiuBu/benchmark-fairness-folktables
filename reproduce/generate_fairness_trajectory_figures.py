"""Race- and sex-based fairness trajectory figures (paper Figs. 2 and 3).

Canonical data convention (see paper, Section III-C):
  * a calendar-year value is the mean over that year's four within-year folds;
  * each sensitive attribute is read from the run configured for that attribute.

Fonts are emitted as TrueType (pdf.fonttype = 42) because IEEE PDF eXpress
rejects Type 3 fonts.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]

# One run per (task, sensitive attribute): a run configured for one attribute also
# emits rows for the other, but those come from a different execution with its own
# seed draws and must not be mixed in (see paper, Section III-C).
RUNS = {
    ("income", "SEX"): ROOT / "results/folktables/benchmark_temporal/folktables_benchmark_temporal/20260507_165711/benchmark_summary_by_year.csv",
    ("income", "RAC1P"): ROOT / "results/folktables/benchmark_temporal_race/folktables_benchmark_temporal_race/20260508_111012/benchmark_summary_by_year.csv",
    ("employment", "SEX"): ROOT / "results/folktables/benchmark_temporal_employment/folktables_benchmark_temporal_employment/20260507_232826/benchmark_summary_by_year.csv",
    ("employment", "RAC1P"): ROOT / "results/folktables/benchmark_temporal_employment_race/folktables_benchmark_temporal_employment_race/20260508_173829/benchmark_summary_by_year.csv",
}

METHOD_COLORS = {
    "baseline": "#264a6e",
    "reweighing": "#b66439",
    "demographic_parity": "#7a5a8a",
    "equalized_odds": "#3f7a6e",
    "fairness_constraint": "#a8434e",
}
METHOD_DISPLAY = {
    "baseline": "Baseline",
    "reweighing": "Reweighing",
    "demographic_parity": "DP post-proc.",
    "equalized_odds": "EO post-proc.",
    "fairness_constraint": "EG reduction",
}
METHOD_ORDER = list(METHOD_DISPLAY)
MAINT_KEEP = [
    ("no-retrain", "Frozen (no-retrain)"),
    ("threshold-recalibration", "Threshold recal."),
    ("last-year-retrain", "Last-year retrain"),
]
GAPS = [("dp_gap_mean", "DP gap"), ("eo_gap_mean", "EO gap")]
YEARS = [2015, 2016, 2017, 2018]

plt.rcParams.update({
    "pdf.fonttype": 42,   # TrueType — required by IEEE PDF eXpress
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
    "grid.color": "#888888",
    "figure.dpi": 200,
})


def render(task: str, attr: str, attr_label: str) -> Path:
    df = pd.read_csv(RUNS[(task, attr)])
    df = df[df["sensitive_attribute_value"].astype(str) == "ALL"]
    df = df[(df["phase"] == "temporal") & (df["sensitive_attribute"] == attr)]
    # the four within-year folds are encoded as YYYY.00/.25/.50/.75, so the
    # calendar year is the floor -- rounding would leak folds into the next year
    df = df.assign(year_int=np.floor(df["year"]).astype(int))
    df = df[df["year_int"].isin(YEARS)]

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.8), sharex=True, sharey="row")
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.10, right=0.78,
                        hspace=0.34, wspace=0.10)

    row_ranges = {}
    for metric, _ in GAPS:
        vals = df[df["method"].isin(METHOD_ORDER)]
        vals = vals[vals["maintenance"].isin([k for k, _ in MAINT_KEEP])][metric].dropna()
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.18 if hi > lo else 0.005
        row_ranges[metric] = (max(0.0, lo - pad), hi + pad)

    for r, (metric, metric_label) in enumerate(GAPS):
        for c, (maint_key, maint_label) in enumerate(MAINT_KEEP):
            ax = axes[r][c]
            sub = df[df["maintenance"] == maint_key]
            for m in METHOD_ORDER:
                s = sub[sub["method"] == m].groupby("year_int", as_index=False)[metric].mean()
                if s.empty:
                    continue
                ax.plot(s["year_int"], s[metric], "-o", color=METHOD_COLORS[m],
                        linewidth=1.2, markersize=2.5, markeredgewidth=0,
                        label=METHOD_DISPLAY[m])
            if r == 0:
                ax.set_title(maint_label, pad=3, color="#222222")
            if c == 0:
                ax.set_ylabel(metric_label)
            ax.set_xticks(YEARS)
            ax.set_xticklabels([str(y) for y in YEARS])
            ax.set_xlim(2014.8, 2018.2)
            ax.set_ylim(*row_ranges[metric])
            ax.tick_params(length=2, width=0.5, color="#666666")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            for spine in ax.spines.values():
                spine.set_color("#666666")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.998, 0.5),
               frameon=False, handlelength=1.4, borderaxespad=0.2)
    fig.suptitle(f"{task.capitalize()} task — {attr_label}", fontsize=10, y=0.97)

    out = Path(__file__).parent / f"fig{2 if attr == 'RAC1P' else 3}_fairness_{attr.lower()}_{task}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for task in ("income", "employment"):
        for attr, label in (("RAC1P", "race (White vs non-White)"), ("SEX", "sex")):
            print("wrote", render(task, attr, label).name)
