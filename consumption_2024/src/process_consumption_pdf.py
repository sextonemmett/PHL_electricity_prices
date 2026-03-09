from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import seaborn as sns
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


REGION_ORDER = ["Luzon", "Visayas", "Mindanao", "Philippines"]
ISLAND_GROUPS = ["Luzon", "Visayas", "Mindanao"]
SECTOR_ORDER = [
    "Residential",
    "Commercial",
    "Industrial",
    "Others",
    "Total Sales",
    "Own-Use",
    "System Loss",
    "Total Consumption",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TABLE_TITLE = "2024 MONTHLY ELECTRICITY SALES and POWER CONSUMPTION by SECTOR"
LINE_SECTORS = ["Residential", "Commercial", "Industrial", "Others"]
STACK_SECTORS = ["Total Sales", "Own-Use", "System Loss"]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.facecolor": "#fcfcfc",
            "figure.facecolor": "white",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "axes.edgecolor": "#bdbdbd",
            "axes.labelcolor": "#2f2f2f",
            "xtick.color": "#2f2f2f",
            "ytick.color": "#2f2f2f",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def get_paths() -> tuple[Path, Path, Path]:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    (output_dir / "png").mkdir(parents=True, exist_ok=True)
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)
    return base_dir, data_dir, output_dir


def find_single_pdf(data_dir: Path) -> Path:
    files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")]
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one PDF in {data_dir}, found {len(files)}: {[f.name for f in files]}")
    return files[0]


def parse_int(value: str | None) -> int:
    if value is None:
        raise ValueError("Encountered missing numeric value in table.")
    cleaned = value.strip().replace(",", "")
    if cleaned == "":
        raise ValueError("Encountered blank numeric value in table.")
    return int(cleaned)


def extract_wide_table(pdf_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, int | str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if len(table) < 3:
                    continue
                if table[0][0] != TABLE_TITLE:
                    continue
                header = table[1]
                if header is None or len(header) < 14:
                    continue
                region = (header[0] or "").strip()
                if region not in REGION_ORDER:
                    continue
                if header[1:13] != MONTHS or (header[13] or "").strip() != "Total":
                    continue

                for raw_row in table[2:]:
                    if raw_row is None or len(raw_row) < 14:
                        continue
                    sector = (raw_row[0] or "").strip()
                    if sector == "":
                        continue

                    values = [parse_int(v) for v in raw_row[1:14]]
                    record: Dict[str, int | str] = {"Region": region, "Sector": sector}
                    for month, value in zip(MONTHS, values[:12]):
                        record[month] = value
                    record["Total"] = values[12]
                    rows.append(record)

    if not rows:
        raise RuntimeError(f"No valid rows were extracted from {pdf_path}.")

    df = pd.DataFrame(rows)
    df["Region"] = pd.Categorical(df["Region"], categories=REGION_ORDER, ordered=True)
    df["Sector"] = pd.Categorical(df["Sector"], categories=SECTOR_ORDER, ordered=True)
    df = df.sort_values(["Region", "Sector"]).reset_index(drop=True)
    return df


def validate_totals(wide_df: pd.DataFrame) -> pd.DataFrame:
    checks: List[Dict[str, int | str]] = []

    row_check = wide_df.copy()
    row_check["Monthly_Sum"] = row_check[MONTHS].sum(axis=1)
    row_check["Difference"] = row_check["Total"] - row_check["Monthly_Sum"]
    for _, row in row_check.iterrows():
        checks.append(
            {
                "Check": "row_monthly_sum_equals_total",
                "Region": str(row["Region"]),
                "Sector": str(row["Sector"]),
                "Difference_MWh": int(row["Difference"]),
            }
        )

    island_df = wide_df[wide_df["Region"].isin(["Luzon", "Visayas", "Mindanao"])].copy()
    philippines_df = wide_df[wide_df["Region"] == "Philippines"].copy()
    island_grouped = island_df.groupby("Sector", as_index=False, observed=False)[MONTHS + ["Total"]].sum()
    merged = island_grouped.merge(philippines_df, on="Sector", suffixes=("_Islands", "_Philippines"))

    for _, row in merged.iterrows():
        for col in MONTHS + ["Total"]:
            diff = int(row[f"{col}_Philippines"] - row[f"{col}_Islands"])
            checks.append(
                {
                    "Check": "philippines_equals_islands_sum",
                    "Region": "Philippines",
                    "Sector": str(row["Sector"]),
                    "Difference_MWh": diff,
                }
            )

    return pd.DataFrame(checks)


def build_long_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    long_df = wide_df.melt(
        id_vars=["Region", "Sector"],
        value_vars=MONTHS,
        var_name="Month",
        value_name="Consumption_MWh",
    )
    long_df.insert(0, "Year", 2024)
    long_df["Month"] = pd.Categorical(long_df["Month"], categories=MONTHS, ordered=True)
    long_df = long_df.sort_values(["Region", "Sector", "Month"]).reset_index(drop=True)
    return long_df


def plot_island_monthly_lines(wide_df: pd.DataFrame, output_dir: Path) -> Path:
    line_df = wide_df[
        wide_df["Region"].isin(ISLAND_GROUPS) & wide_df["Sector"].isin(LINE_SECTORS)
    ].copy()
    line_df["Region"] = pd.Categorical(line_df["Region"], categories=ISLAND_GROUPS, ordered=True)
    line_df["Sector"] = pd.Categorical(line_df["Sector"], categories=LINE_SECTORS, ordered=True)
    line_df = line_df.sort_values(["Region", "Sector"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    line_palette = sns.color_palette("muted", n_colors=len(LINE_SECTORS))
    colors = dict(zip(LINE_SECTORS, line_palette))

    luzon_max = float(
        line_df[line_df["Region"] == "Luzon"][MONTHS].to_numpy().max()
    )
    vismin_max = float(
        line_df[line_df["Region"].isin(["Visayas", "Mindanao"])][MONTHS].to_numpy().max()
    )
    luzon_ylim = (0, luzon_max * 1.1)
    vismin_ylim = (0, vismin_max * 1.1)

    for i, region in enumerate(ISLAND_GROUPS):
        ax = axes[i]
        region_slice = line_df[line_df["Region"] == region]
        for sector in LINE_SECTORS:
            sector_row = region_slice[region_slice["Sector"] == sector]
            if sector_row.empty:
                continue
            y = sector_row.iloc[0][MONTHS].to_list()
            ax.plot(MONTHS, y, marker="o", linewidth=2.0, markersize=4, color=colors[sector], label=sector)
        ax.set_title(region, weight="bold")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Consumption (MWh)")
        if region == "Luzon":
            ax.set_ylim(luzon_ylim)
        else:
            ax.set_ylim(vismin_ylim)

    axes[-1].set_xlabel("Month")
    axes[-1].tick_params(axis="x", rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle("Monthly Consumption by Sector and Island Group (2024)", y=1.02, fontsize=15, weight="bold")
    fig.tight_layout()

    out_path = output_dir / "png" / "island_sector_monthly_lines_shared_y.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_island_sector_share_stacked(wide_df: pd.DataFrame, output_dir: Path) -> Path:
    share_df = wide_df[
        wide_df["Region"].isin(ISLAND_GROUPS) & wide_df["Sector"].isin(LINE_SECTORS)
    ][["Region", "Sector", "Total"]].copy()
    share_df["Region"] = pd.Categorical(share_df["Region"], categories=ISLAND_GROUPS, ordered=True)
    share_df["Sector"] = pd.Categorical(share_df["Sector"], categories=LINE_SECTORS, ordered=True)

    pivot = (
        share_df.pivot(index="Region", columns="Sector", values="Total")
        .reindex(index=ISLAND_GROUPS, columns=LINE_SECTORS)
        .fillna(0)
    )
    shares = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("muted", n_colors=len(LINE_SECTORS))
    shares.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.65)
    ax.set_title("Sector Share of Island Group Total Sales (2024)", pad=12, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Island Group Total")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="x", visible=False)
    ax.legend(
        title="",
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )
    fig.tight_layout()

    out_path = output_dir / "png" / "island_sector_share_stacked_100pct_bar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_island_sales_ownuse_loss_stacked(wide_df: pd.DataFrame, output_dir: Path) -> Path:
    stack_df = wide_df[
        wide_df["Region"].isin(ISLAND_GROUPS) & wide_df["Sector"].isin(STACK_SECTORS)
    ][["Region", "Sector", "Total"]].copy()
    stack_df["Region"] = pd.Categorical(stack_df["Region"], categories=ISLAND_GROUPS, ordered=True)
    stack_df["Sector"] = pd.Categorical(stack_df["Sector"], categories=STACK_SECTORS, ordered=True)

    pivot = (
        stack_df.pivot(index="Region", columns="Sector", values="Total")
        .reindex(index=ISLAND_GROUPS, columns=STACK_SECTORS)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("muted", n_colors=len(STACK_SECTORS))
    pivot.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.65)
    ax.set_title("2024 Total Sales, Own-Use, and System Loss by Island Group", pad=12, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Consumption (MWh)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="x", visible=False)
    ax.legend(title="", frameon=False)
    fig.tight_layout()

    out_path = output_dir / "png" / "island_sales_ownuse_systemloss_stacked_bar.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    setup_style()
    _, data_dir, output_dir = get_paths()
    pdf_path = find_single_pdf(data_dir)
    wide_df = extract_wide_table(pdf_path)
    checks_df = validate_totals(wide_df)
    long_df = build_long_table(wide_df)

    wide_output = output_dir / "csv" / "monthly_power_consumption_wide.csv"
    long_output = output_dir / "csv" / "monthly_power_consumption_long.csv"
    checks_output = output_dir / "csv" / "monthly_power_consumption_validation_checks.csv"
    lines_plot_output = plot_island_monthly_lines(wide_df, output_dir)
    sector_share_plot_output = plot_island_sector_share_stacked(wide_df, output_dir)
    stacked_plot_output = plot_island_sales_ownuse_loss_stacked(wide_df, output_dir)

    wide_df.to_csv(wide_output, index=False)
    long_df.to_csv(long_output, index=False)
    checks_df.to_csv(checks_output, index=False)

    max_abs_diff = checks_df["Difference_MWh"].abs().max()
    print(f"Processed: {pdf_path.name}")
    print(f"Wrote: {wide_output}")
    print(f"Wrote: {long_output}")
    print(f"Wrote: {checks_output}")
    print(f"Wrote: {lines_plot_output}")
    print(f"Wrote: {sector_share_plot_output}")
    print(f"Wrote: {stacked_plot_output}")
    print(f"Max absolute validation difference (MWh): {int(max_abs_diff)}")


if __name__ == "__main__":
    main()
