"""Study-design figure (paper Fig. 1) — two panels.

  * Panel A: temporal protocol (2014 development cohort, 2015-2018 evaluation
    years, four within-year folds aggregated to calendar-year summaries).
  * Panel B: the experimental design space as a 5x5 grid, with the fairness
    pipelines and the maintenance policies as co-equal axes, plus the remaining
    factors (seeds, tasks, sensitive attributes) that multiply out to the
    2,000 model evaluations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --- palette (muted, print-friendly; consistent with the result figures) ---
INK        = "#222222"
GREY       = "#555555"
GREY_LIGHT = "#8a8a8a"
RULE       = "#666666"
BLUE       = "#264a6e"
BLUE_FILL  = "#eaf0f6"
WARM       = "#b66439"
WARM_FILL  = "#fbf0e8"
CELL       = "#3f6f92"

plt.rcParams.update({
    "pdf.fonttype": 42,   # TrueType — required by IEEE PDF eXpress (Type 3 is rejected)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times", "Times New Roman", "DejaVu Serif"],
    "figure.dpi": 200,
})

PIPELINES = ["Baseline", "Reweighing", "DP post-proc.", "EO post-proc.", "EG reduction"]
POLICIES  = ["Frozen", "Threshold\nrecal.", "Last-year\nretrain", "Cumulative\nretrain", "Cumulative\nfixed-$n$"]
YEARS     = [2015, 2016, 2017, 2018]


def panel_label(ax, x, y, letter):
    ax.text(x, y, letter, fontsize=11, fontweight="bold", color=INK,
            ha="left", va="top", transform=ax.transAxes)


def build():
    fig = plt.figure(figsize=(8.4, 3.72))
    # Panel A occupies the top ~34%, panel B the rest.
    axA = fig.add_axes([0.052, 0.640, 0.905, 0.300])
    axB = fig.add_axes([0.052, 0.040, 0.905, 0.545])
    for ax in (axA, axB):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

    # ============================ PANEL A ============================
    panel_label(axA, -0.045, 1.10, "A")
    axA.text(4.5, 108, "Temporal protocol", fontsize=11, fontweight="bold",
             color=BLUE, ha="left", va="top")

    # development cohort box
    devbox = FancyBboxPatch((1.5, 26), 24, 40,
                            boxstyle="round,pad=1.2,rounding_size=2",
                            linewidth=1.0, facecolor=BLUE_FILL, edgecolor=BLUE)
    axA.add_patch(devbox)
    axA.text(13.5, 57, "2014 development cohort", ha="center", va="center",
             fontsize=8.6, fontweight="bold", color=INK)
    axA.text(13.5, 42, "stratified split\ntrain 70 / val 10 / hold-out 20",
             ha="center", va="center", fontsize=7.6, color=GREY)

    # arrow into the evaluation years
    axA.add_patch(FancyArrowPatch((26.5, 46), (34.5, 46), arrowstyle="-|>",
                                  mutation_scale=11, linewidth=1.1, color=RULE))
    axA.text(30.5, 52, "freeze or\nupdate", ha="center", va="bottom",
             fontsize=6.6, color=GREY_LIGHT, style="italic")

    # evaluation years, each drawn as four fold ticks
    x0, span = 37.0, 60.0
    step = span / len(YEARS)
    for k, yr in enumerate(YEARS):
        xa = x0 + k * step
        axA.add_patch(Rectangle((xa, 30), step - 3.0, 32, linewidth=0.9,
                                facecolor=WARM_FILL, edgecolor=WARM))
        axA.text(xa + (step - 3.0) / 2, 70, str(yr), ha="center", va="center",
                 fontsize=8.4, fontweight="bold", color=INK)
        # four within-year folds, laid out strictly inside the year box
        box_w, margin, gap = step - 3.0, 1.5, 0.8
        fold_w = (box_w - 2 * margin - 3 * gap) / 4
        for j in range(4):
            fx = xa + margin + j * (fold_w + gap)
            axA.add_patch(Rectangle((fx, 38), fold_w, 17, linewidth=0,
                                    facecolor=WARM, alpha=0.75))
    # brace-ish rule under the years + explanation
    axA.plot([x0, x0 + span - 3.0], [23, 23], color=RULE, linewidth=0.8)
    axA.text(x0 + (span - 3.0) / 2, 16,
             "4 stratified within-year folds per year, aggregated to calendar-year summaries "
             "(folds are diagnostic strata, not calendar quarters)",
             ha="center", va="top", fontsize=6.9, color=GREY)

    # ============================ PANEL B ============================
    panel_label(axB, -0.045, 1.06, "B")
    axB.text(4.5, 100, "Experimental design space", fontsize=11, fontweight="bold",
             color=BLUE, ha="left", va="top")

    grid_left, grid_right = 21.0, 62.0
    grid_top, grid_bot = 76.0, 20.0
    ncol, nrow = len(POLICIES), len(PIPELINES)
    cw = (grid_right - grid_left) / ncol
    ch = (grid_top - grid_bot) / nrow

    # column axis label + headers
    axB.text((grid_left + grid_right) / 2, 92, "MAINTENANCE POLICY",
             ha="center", va="center", fontsize=7.6, fontweight="bold", color=GREY)
    for j, pol in enumerate(POLICIES):
        axB.text(grid_left + (j + 0.5) * cw, 84.0, pol, ha="center", va="center",
                 fontsize=6.9, color=INK, linespacing=1.25)

    # row axis label + headers
    axB.text(1.5, (grid_top + grid_bot) / 2, "FAIRNESS PIPELINE",
             ha="center", va="center", fontsize=7.6, fontweight="bold",
             color=GREY, rotation=90)
    for i, pipe in enumerate(PIPELINES):
        axB.text(grid_left - 1.6, grid_top - (i + 0.5) * ch, pipe,
                 ha="right", va="center", fontsize=7.0, color=INK)

    # the cells
    for i in range(nrow):
        for j in range(ncol):
            cx = grid_left + (j + 0.5) * cw
            cy = grid_top - (i + 0.5) * ch
            axB.add_patch(Rectangle((cx - cw * 0.40, cy - ch * 0.34),
                                    cw * 0.80, ch * 0.68,
                                    linewidth=0.8, facecolor="#dbe5ef",
                                    edgecolor="#7f9ab3"))
            axB.plot([cx], [cy], marker="s", markersize=3.4,
                     color=CELL, markeredgewidth=0)

    # multiplication annotation to the right of the grid
    ax_x = grid_right + 4.0
    axB.text(ax_x, 71, "each cell", fontsize=7.2, color=GREY, ha="left", va="center")
    axB.text(ax_x, 64, "20 random seeds", fontsize=8.0, color=INK,
             fontweight="bold", ha="left", va="center")
    axB.plot([ax_x, ax_x + 26], [58, 58], color="#cfcfcf", linewidth=0.7)
    axB.text(ax_x, 51, "$\\times$  2 prediction tasks", fontsize=7.6, color=INK, ha="left", va="center")
    axB.text(ax_x, 44, "$\\times$  2 sensitive attributes", fontsize=7.6, color=INK, ha="left", va="center")
    axB.plot([ax_x, ax_x + 26], [38, 38], color="#cfcfcf", linewidth=0.7)
    axB.text(ax_x, 30, "2,000 model", fontsize=8.6, fontweight="bold", color=WARM, ha="left", va="center")
    axB.text(ax_x, 23, "evaluations", fontsize=8.6, fontweight="bold", color=WARM, ha="left", va="center")

    # metric strip under the grid
    axB.plot([grid_left - 1.0, grid_right], [13, 13], color=RULE, linewidth=0.8)
    axB.text((grid_left + grid_right) / 2 - 1.0, 6.5,
             "reported per cell:  DP gap  ·  EO gap  ·  O/E gap  ·  PPR  ·  accuracy, F1, AUC, Brier",
             ha="center", va="center", fontsize=6.9, color=GREY)

    out = Path(__file__).parent / "fig1_study_design.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")
