"""
visualisations.py
=================
Standalone figure generator for:
  "A Predictive Supply Chain Framework: Leveraging Small Language Models
   for Sentiment-Based Demand Forecasting"

Reproduces all four paper figures from pre-computed results — no GPU or
SLM required. Useful for quickly regenerating plots without re-running
the full notebook pipeline.

Usage
-----
    python visualisations.py                  # generates all figures
    python visualisations.py --fig 4          # Fig 4 only (temporal aggregation)
    python visualisations.py --fig 2          # Fig 2/3 only (ablation bar + forecast)
    python visualisations.py --fig 5          # Fig 5 only (domain sensitivity)
    python visualisations.py --fig dist       # Extended: score distribution

Output
------
    figures/fig2_ablation_results.png
    figures/fig4_temporal_aggregation.png
    figures/fig5_domain_sensitivity.png
    figures/fig_score_distribution.png

Paper Results (hard-coded from Table I)
----------------------------------------
    All Beauty  — Baseline MAE: 4.59 | Multimodal MAE: 4.46 | Δ: −2.83%
    Electronics — Baseline MAE: 3.20 | Multimodal MAE: 3.31 | Δ: +3.44%
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paper results (Table I) ───────────────────────────────────────────────────
RESULTS = {
    "All Beauty":  {"mae_base": 4.59, "mae_multi": 4.46, "improvement":  2.83},
    "Electronics": {"mae_base": 3.20, "mae_multi": 3.31, "improvement": -3.44},
}

# ── Shared style ──────────────────────────────────────────────────────────────
PALETTE = {
    "baseline":   "#A0A0A0",
    "multimodal": "#2E86C1",
    "actual":     "#1A1A1A",
    "beauty":     "#2E86C1",
    "elec":       "#E74C3C",
}
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "figure.dpi":         150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Temporal Aggregation as Low-Pass Filter
# ─────────────────────────────────────────────────────────────────────────────
def fig_temporal_aggregation(save=True):
    """
    Fig 4 (Paper §II-D): Illustrative comparison of daily vs weekly granularity.
    Uses synthetic data (np.random.seed(42)) — identical to the notebook cell.
    """
    np.random.seed(42)
    days         = np.arange(1, 101)
    daily_sales  = 5 + 0.05 * days + np.random.poisson(3, 100)
    weekly_sales = [np.sum(daily_sales[i:i + 7]) for i in range(0, 100, 7)]
    weekly_days  = np.arange(1, len(weekly_sales) + 1) * 7

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Temporal Aggregation as a Low-Pass Filter  (Paper Fig. 4 — Illustrative)",
        fontsize=13, y=1.02
    )

    # Daily — noisy
    axes[0].bar(days, daily_sales, color="lightgray", alpha=0.7)
    axes[0].plot(days, daily_sales, color="red", linewidth=1, alpha=0.6)
    axes[0].set_title("Daily Granularity: High Stochastic Noise\n(Result: 0% improvement)", fontsize=12)
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Units Sold")

    # Weekly — smoothed
    axes[1].step(weekly_days, weekly_sales, where="mid", color="steelblue", linewidth=2.5)
    axes[1].fill_between(weekly_days, weekly_sales, step="mid", alpha=0.15, color="steelblue")
    axes[1].set_title("Weekly Granularity: Signal Smoothing\n(Result: 2.83% improvement)", fontsize=12)
    axes[1].set_xlabel("Weeks (Aggregated)")
    axes[1].set_ylabel("Total Weekly Units Sold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_temporal_aggregation.png")
    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 & 3 — Ablation Study Results
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation_results(save=True):
    """
    Fig 2 & 3 (Paper §III): MAE bar chart + illustrative forecast comparison.
    Uses paper Table I values. Forecast curve uses synthetic illustrative data.
    """
    categories   = list(RESULTS.keys())
    mae_base_all = [RESULTS[c]["mae_base"]  for c in categories]
    mae_mult_all = [RESULTS[c]["mae_multi"] for c in categories]
    improvements = [RESULTS[c]["improvement"] for c in categories]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Study: Impact of SLM Sentiment on Forecast Error", fontsize=14)

    # ── Left: MAE grouped bar chart ──────────────────────────────────────────
    x, width = np.arange(len(categories)), 0.35
    axes[0].bar(x - width / 2, mae_base_all, width,
                label="Baseline (Sales Only)", color=PALETTE["baseline"], alpha=0.85)
    axes[0].bar(x + width / 2, mae_mult_all, width,
                label="Multimodal (Sales + SLM)", color=PALETTE["multimodal"], alpha=0.85)

    for i, (imp, xpos) in enumerate(zip(improvements, x)):
        colour = "green" if imp > 0 else "red"
        label  = f"{imp:+.2f}%"
        ypos   = max(mae_base_all[i], mae_mult_all[i]) + 0.08
        axes[0].text(xpos, ypos, label,
                     ha="center", color=colour, fontweight="bold", fontsize=12)

    axes[0].set_xlabel("Product Category")
    axes[0].set_ylabel("Mean Absolute Error (lower is better)")
    axes[0].set_title("MAE: Baseline vs Multimodal  (Paper Fig. 2 & 3)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].set_ylim(0, max(mae_base_all) * 1.4)

    # ── Right: Illustrative forecast curve (Beauty category) ─────────────────
    np.random.seed(7)
    n = 50
    t = np.arange(n)
    actual    = np.clip(10 + np.cumsum(np.random.randn(n) * 0.4), 2, 25).astype(float)
    pred_base = actual + np.random.randn(n) * RESULTS["All Beauty"]["mae_base"]
    pred_mult = actual + np.random.randn(n) * RESULTS["All Beauty"]["mae_multi"]

    axes[1].plot(t, actual,    label="Actual Demand",
                 color=PALETTE["actual"], linewidth=2, marker="o", markersize=3)
    axes[1].plot(t, pred_base, label=f'Baseline  (MAE: {RESULTS["All Beauty"]["mae_base"]:.4f})',
                 color="gray", linewidth=1.5, linestyle="--", alpha=0.8)
    axes[1].plot(t, pred_mult, label=f'Multimodal (MAE: {RESULTS["All Beauty"]["mae_multi"]:.4f})',
                 color=PALETTE["multimodal"], linewidth=2)

    axes[1].text(0.02, 0.95,
                 f'Error Reduction: {RESULTS["All Beauty"]["improvement"]:+.2f}%',
                 transform=axes[1].transAxes, fontsize=11, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    axes[1].set_title("Forecast Accuracy: All Beauty — Weekly  (Illustrative)")
    axes[1].set_xlabel("Test Samples (Weekly Time Steps)")
    axes[1].set_ylabel("Demand Volume")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_ablation_results.png")
    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Domain Sensitivity (Category Attributes Table)
# ─────────────────────────────────────────────────────────────────────────────
def fig_domain_sensitivity(save=True):
    """
    Fig 5 (Paper §IV): Hedonic vs utilitarian product category comparison.
    Visualises the category attribute table from the paper as a styled heatmap.
    """
    attributes = [
        "Product type",
        "Description style",
        "Sentiment utility",
        "Observed MAE lift",
    ]
    beauty_vals = [
        "Hedonic / Subjective",
        "Affective / Emotional",
        "High predictive signal",
        "+2.83% improvement",
    ]
    elec_vals = [
        "Utilitarian / Objective",
        "Technical / Functional",
        "Low signal / Noise",
        "−3.44% accuracy drop",
    ]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    fig.suptitle("Category Attributes: Domain Sensitivity Analysis  (Paper Fig. 5)",
                 fontsize=13, y=1.02)

    col_labels = ["Attribute", "All Beauty", "Electronics"]
    table_data = [[a, b, e] for a, b, e in zip(attributes, beauty_vals, elec_vals)]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.28, 0.36, 0.36],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Header styling
    for col in range(3):
        tbl[(0, col)].set_facecolor("#2E86C1")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    # Row colours
    for row in range(1, len(attributes) + 1):
        tbl[(row, 0)].set_facecolor("#F2F3F4")
        tbl[(row, 0)].set_text_props(fontweight="bold")
        tbl[(row, 1)].set_facecolor("#EBF5FB")
        tbl[(row, 2)].set_facecolor("#FDEDEC")

    # Highlight the result row
    tbl[(4, 1)].set_text_props(color="green", fontweight="bold")
    tbl[(4, 2)].set_text_props(color="red",   fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_domain_sensitivity.png")
    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Extended — Score Distribution (Section 9 of notebook)
# ─────────────────────────────────────────────────────────────────────────────
def fig_score_distribution(save=True):
    """
    Extended Fig (Notebook §9.2): SLM Market Appeal Index distribution.
    Uses illustrative synthetic score-demand relationship matching paper correlations:
      All Beauty:  Pearson r ≈ +0.04  (mild positive)
      Electronics: Pearson r ≈ +0.02  (near-zero)
    """
    np.random.seed(42)
    scores = np.arange(1, 11)

    # Simulate score distribution and mean demand per score
    beauty_counts  = np.array([5, 8, 15, 22, 35, 40, 38, 28, 12, 4])   # peaks at 6-7
    elec_counts    = np.array([3, 6, 12, 20, 38, 42, 36, 24, 14, 5])   # similar distribution
    beauty_demand  = 6 + 0.3 * scores + np.random.randn(10) * 0.8      # gentle positive slope
    elec_demand    = 8 + 0.1 * scores + np.random.randn(10) * 1.5      # nearly flat

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Market Appeal Index Distribution by Category  (Notebook §9.2)", fontsize=13)

    for ax, label, counts, demand, corr in [
        (axes[0], "All Beauty",  beauty_counts, beauty_demand,  0.0043),
        (axes[1], "Electronics", elec_counts,   elec_demand,    0.0224),
    ]:
        ax.bar(scores, counts, color=PALETTE["multimodal"], alpha=0.7, edgecolor="white")
        ax.set_ylabel("Number of Product-Weeks", color=PALETTE["multimodal"])
        ax.tick_params(axis="y", labelcolor=PALETTE["multimodal"])

        ax2 = ax.twinx()
        ax2.plot(scores, demand, color=PALETTE["elec"], marker="o",
                 linewidth=2, label="Mean weekly demand")
        ax2.set_ylabel("Mean Weekly Demand", color=PALETTE["elec"], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=PALETTE["elec"])

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Market Appeal Score (1–10)")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.text(0.03, 0.95, f"Pearson r = {corr:.4f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_score_distribution.png")
    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures for the Supply Chain SLM framework."
    )
    parser.add_argument(
        "--fig",
        choices=["2", "4", "5", "dist", "all"],
        default="all",
        help=(
            "Which figure to generate: "
            "2=ablation results, 4=temporal aggregation, "
            "5=domain sensitivity, dist=score distribution, all=all figures (default)"
        ),
    )
    args = parser.parse_args()

    fig_map = {
        "4":    fig_temporal_aggregation,
        "2":    fig_ablation_results,
        "5":    fig_domain_sensitivity,
        "dist": fig_score_distribution,
    }

    print(f"Generating figures → ./{OUT_DIR}/")
    if args.fig == "all":
        for name, fn in fig_map.items():
            print(f"\n[Fig {name}]")
            fn()
    else:
        print(f"\n[Fig {args.fig}]")
        fig_map[args.fig]()

    print("\nDone.")


if __name__ == "__main__":
    main()
