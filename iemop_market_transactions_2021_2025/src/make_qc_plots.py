from __future__ import annotations

from pathlib import Path
from math import ceil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_ORDER = [
    "Metered_Quantity_GWh",
    "Bilateral_Quantity_GWh",
    "Spot_Quantity_GWh",
    "Daily_Average_MQ_GWh",
    "Customer_ESSP_PHP_per_kWh",
    "LuzVis_ESSP_PHP_per_kWh",
    "Mindanao_ESSP_PHP_per_kWh",
    "LVM_ESSP_PHP_per_kWh",
    "ESSP_PHP_per_kWh",
    "Trading_Amount_Billion_PHP",
]

METRIC_LABELS = {
    "Metered_Quantity_GWh": "Metered Quantity (GWh)",
    "Bilateral_Quantity_GWh": "Bilateral Quantity (GWh)",
    "Spot_Quantity_GWh": "Spot Quantity (GWh)",
    "Daily_Average_MQ_GWh": "Daily Average MQ (GWh)",
    "Customer_ESSP_PHP_per_kWh": "Customer ESSP (PhP/kWh)",
    "LuzVis_ESSP_PHP_per_kWh": "Luz-Vis ESSP (PhP/kWh)",
    "Mindanao_ESSP_PHP_per_kWh": "Mindanao ESSP (PhP/kWh)",
    "LVM_ESSP_PHP_per_kWh": "LVM ESSP (PhP/kWh)",
    "ESSP_PHP_per_kWh": "ESSP (PhP/kWh)",
    "Trading_Amount_Billion_PHP": "Trading Amount (Billion PHP)",
}

SCOPE_LABELS = {
    "Luzon_Visayas": "Luzon + Visayas",
    "Luzon_Visayas_Mindanao": "Luzon + Visayas + Mindanao",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e2e2e2",
            "grid.linewidth": 0.7,
            "font.size": 10,
        }
    )


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(base_dir / "outputs" / "csv" / "market_transactions_monthly_latest.csv")
    overlap = pd.read_csv(base_dir / "outputs" / "csv" / "market_transactions_overlap_checks.csv")
    monthly["billing_month"] = pd.to_datetime(monthly["billing_month"])
    monthly["report_month"] = pd.to_datetime(monthly["report_month"])
    overlap["billing_month"] = pd.to_datetime(overlap["billing_month"])
    return monthly, overlap


def make_metric_small_multiples(monthly: pd.DataFrame, output_path: Path) -> None:
    metrics = [metric for metric in METRIC_ORDER if metric in set(monthly["metric"])]
    ncols = 2
    nrows = ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(12, 3.2 * nrows)), sharex=True)
    axes_flat = axes.flatten()

    palette = {"Luzon_Visayas": "#1f77b4", "Luzon_Visayas_Mindanao": "#d62728"}

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        subset = monthly[monthly["metric"] == metric].sort_values("billing_month")
        for scope, group in subset.groupby("market_scope"):
            color = palette.get(scope, "#4d4d4d")
            ax.plot(
                group["billing_month"],
                group["value"],
                marker="o",
                markersize=3.0,
                linewidth=1.8,
                label=SCOPE_LABELS.get(scope, scope),
                color=color,
            )
            overlap_pts = group[group["source_report_count"] > 1]
            if not overlap_pts.empty:
                ax.scatter(
                    overlap_pts["billing_month"],
                    overlap_pts["value"],
                    s=28,
                    facecolors="none",
                    edgecolors=color,
                    linewidth=1.0,
                )

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11, weight="bold")
        ax.set_ylabel("Value")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    for idx in range(len(metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Market Transactions QC: Final Monthly Series", fontsize=17, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_balance_plot(monthly: pd.DataFrame, output_path: Path) -> None:
    core = monthly[monthly["metric"].isin(["Metered_Quantity_GWh", "Bilateral_Quantity_GWh", "Spot_Quantity_GWh"])].copy()
    scopes = [scope for scope in ["Luzon_Visayas", "Luzon_Visayas_Mindanao"] if scope in set(core["market_scope"])]
    fig, axes = plt.subplots(len(scopes), 1, figsize=(14, max(5, 3.5 * len(scopes))), sharex=True)
    if len(scopes) == 1:
        axes = [axes]

    for ax, scope in zip(axes, scopes):
        scope_df = core[core["market_scope"] == scope].pivot_table(
            index="billing_month", columns="metric", values="value", aggfunc="last"
        )
        scope_df = scope_df.sort_index()
        scope_df["balance_diff"] = scope_df["Metered_Quantity_GWh"] - (
            scope_df["Bilateral_Quantity_GWh"] + scope_df["Spot_Quantity_GWh"]
        )
        ax.plot(scope_df.index, scope_df["balance_diff"], marker="o", linewidth=1.8, color="#2a9d8f")
        ax.axhline(0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_title(f"Balance Check: Metered - (Bilateral + Spot) | {SCOPE_LABELS.get(scope, scope)}", fontsize=11)
        ax.set_ylabel("Difference (GWh)")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_overlap_heatmap(overlap: pd.DataFrame, output_path: Path) -> None:
    overlap = overlap.copy()
    overlap["row_label"] = overlap["metric"].map(METRIC_LABELS).fillna(overlap["metric"]) + " | " + overlap[
        "market_scope"
    ].map(SCOPE_LABELS).fillna(overlap["market_scope"])
    matrix = overlap.pivot_table(
        index="row_label",
        columns="billing_month",
        values="absolute_difference",
        aggfunc="max",
        fill_value=0.0,
    ).sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    vmax = max(1.0, float(matrix.values.max()))

    fig, ax = plt.subplots(figsize=(18, max(6, 0.5 * len(matrix.index))))
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, cbar_kws={"label": "Absolute Difference"})
    ax.set_title("Overlap Check Heatmap (Absolute Difference)", fontsize=14, weight="bold")
    ax.set_xlabel("Billing Month")
    ax.set_ylabel("Metric | Scope")
    tick_idx = list(range(0, len(matrix.columns), 3))
    ax.set_xticks([idx + 0.5 for idx in tick_idx])
    ax.set_xticklabels([matrix.columns[idx].strftime("%Y-%m") for idx in tick_idx], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_overlap_top_discrepancies(overlap: pd.DataFrame, output_path: Path) -> None:
    non_zero = overlap[overlap["absolute_difference"] > 0].copy()
    non_zero = non_zero.sort_values("absolute_difference", ascending=False).head(20)
    if non_zero.empty:
        non_zero = overlap.sort_values("absolute_difference", ascending=False).head(20)

    labels = []
    for _, row in non_zero.iterrows():
        metric = METRIC_LABELS.get(row["metric"], row["metric"])
        scope = "LV" if row["market_scope"] == "Luzon_Visayas" else "LVM"
        labels.append(f"{row['billing_month'].strftime('%Y-%m')} | {scope} | {metric}")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.barh(range(len(non_zero)), non_zero["absolute_difference"].values, color="#e76f51")
    ax.set_yticks(range(len(non_zero)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Difference")
    ax.set_title("Top 20 Overlap Discrepancies", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_six_line_quantities_plot(monthly: pd.DataFrame, output_path: Path) -> None:
    metrics = ["Metered_Quantity_GWh", "Bilateral_Quantity_GWh", "Spot_Quantity_GWh"]
    subset = monthly[monthly["metric"].isin(metrics)].copy()
    subset = subset[subset["market_scope"].isin(["Luzon_Visayas", "Luzon_Visayas_Mindanao"])]

    pivot = subset.pivot_table(
        index=["billing_month", "market_scope"],
        columns="metric",
        values="value",
        aggfunc="last",
    ).reset_index()

    blue_palette = {
        "Metered_Quantity_GWh": "#08306b",
        "Bilateral_Quantity_GWh": "#2171b5",
        "Spot_Quantity_GWh": "#6baed6",
    }
    red_palette = {
        "Metered_Quantity_GWh": "#67000d",
        "Bilateral_Quantity_GWh": "#cb181d",
        "Spot_Quantity_GWh": "#fb6a4a",
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    for scope in ["Luzon_Visayas", "Luzon_Visayas_Mindanao"]:
        scope_df = pivot[pivot["market_scope"] == scope].sort_values("billing_month")
        for metric in metrics:
            color = blue_palette[metric] if scope == "Luzon_Visayas" else red_palette[metric]
            label = f"{SCOPE_LABELS[scope]} - {METRIC_LABELS[metric]}"
            ax.plot(
                scope_df["billing_month"],
                scope_df[metric],
                linewidth=2.0,
                marker="o",
                markersize=3.0,
                color=color,
                label=label,
            )

    ax.set_title("Metered, Bilateral, Spot: Luzon+Visayas vs Three Island Groups", fontsize=14, weight="bold")
    ax.set_ylabel("GWh")
    ax.set_xlabel("Billing Month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    base_dir = Path(__file__).resolve().parents[1]
    output_png = base_dir / "outputs" / "png"
    output_png.mkdir(parents=True, exist_ok=True)

    monthly, overlap = load_data(base_dir)

    make_metric_small_multiples(monthly, output_png / "qc_metric_small_multiples.png")
    make_balance_plot(monthly, output_png / "qc_energy_balance_by_scope.png")
    make_six_line_quantities_plot(monthly, output_png / "qc_6line_quantities_lv_vs_lvm.png")
    make_overlap_heatmap(overlap, output_png / "qc_overlap_heatmap.png")
    make_overlap_top_discrepancies(overlap, output_png / "qc_top_overlap_discrepancies.png")

    print(f"Wrote: {output_png / 'qc_metric_small_multiples.png'}")
    print(f"Wrote: {output_png / 'qc_energy_balance_by_scope.png'}")
    print(f"Wrote: {output_png / 'qc_6line_quantities_lv_vs_lvm.png'}")
    print(f"Wrote: {output_png / 'qc_overlap_heatmap.png'}")
    print(f"Wrote: {output_png / 'qc_top_overlap_discrepancies.png'}")


if __name__ == "__main__":
    main()
