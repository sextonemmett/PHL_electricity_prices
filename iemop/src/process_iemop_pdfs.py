from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pdfplumber
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter


MARKET_METRICS = [
    ("Metered Quantity (GWh)", "Metered_Quantity_GWh"),
    ("Bilateral Quantity (GWh)", "Bilateral_Quantity_GWh"),
    ("Spot Quantity (GWh)", "Spot_Quantity_GWh"),
    ("Daily Average MQ1 (GWh)", "Daily_Average_MQ_GWh"),
    ("ESSP2 (PhP/KWh)", "ESSP_PHP_per_kWh"),
    ("Trading Amount", "Trading_Amount_Billion_PHP"),
]
ISLANDS = ["Luzon", "Visayas", "Mindanao"]
MONTH_TOKEN_RE = re.compile(r"\d{2}-[A-Za-z]{3}$")
DECIMAL_RE = re.compile(r"\d+\.\d{2}")


@dataclass
class PricesPageParse:
    page: int
    first_month: pd.Timestamp
    second_month: pd.Timestamp
    first_values: Dict[str, float]
    second_values: Dict[str, float]


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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def month_token_to_timestamp(token: str) -> pd.Timestamp:
    return pd.to_datetime(token, format="%y-%b").to_period("M").to_timestamp()


def month_label_to_timestamp(label: str) -> pd.Timestamp:
    return pd.to_datetime(label, errors="raise").to_period("M").to_timestamp()


def parse_numeric_line(line: str) -> List[float]:
    vals: List[float] = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", line):
        cleaned = token.replace(",", "")
        if cleaned in {"", "-", "."}:
            continue
        vals.append(float(cleaned))
    return vals


def extract_market_rows(pdf_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            lines = [(page.extract_text() or "").splitlines()]
            if not lines[0]:
                continue
            page_lines = [ln.strip() for ln in lines[0] if ln and ln.strip()]

            billing_line = next((ln for ln in page_lines if ln.startswith("BILLING PERIOD")), None)
            if billing_line is None:
                continue
            month_tokens = [tok for tok in billing_line.replace("BILLING PERIOD", "").split() if MONTH_TOKEN_RE.match(tok)]
            if not month_tokens:
                continue
            months = [month_token_to_timestamp(tok) for tok in month_tokens]

            metric_to_vals: Dict[str, List[float]] = {}
            for label, metric in MARKET_METRICS:
                label_idx = next((i for i, ln in enumerate(page_lines) if ln.startswith(label)), None)
                if label_idx is None:
                    continue

                label_line = page_lines[label_idx]
                tail = label_line[len(label) :].strip()
                vals = parse_numeric_line(tail)

                if len(vals) != len(months):
                    candidates: List[float] = []
                    for offset in (1, 2):
                        idx = label_idx + offset
                        if idx >= len(page_lines):
                            break
                        candidates.extend(parse_numeric_line(page_lines[idx]))
                        if len(candidates) >= len(months):
                            break
                    vals = candidates[: len(months)]

                if len(vals) != len(months):
                    continue

                metric_to_vals[metric] = vals

            if len(metric_to_vals) != len(MARKET_METRICS):
                raise RuntimeError(f"Could not parse all market metrics on page {page_idx} of {pdf_path.name}")

            for month_pos, month in enumerate(months):
                for _, metric in MARKET_METRICS:
                    rows.append(
                        {
                            "Page": page_idx,
                            "Billing_Month": month,
                            "Metric": metric,
                            "Value": metric_to_vals[metric][month_pos],
                        }
                    )

    if not rows:
        raise RuntimeError(f"No market rows parsed from {pdf_path}")
    return pd.DataFrame(rows)


def _extract_luzon_values(lines: List[str], visayas_slide_idx: int) -> tuple[float, float]:
    header_idx = next(i for i, line in enumerate(lines) if "NORTH LUZON" in line and "SOUTH LUZON" in line)
    block = lines[header_idx:visayas_slide_idx]

    first_value = None
    for line in block:
        clean_decimals = DECIMAL_RE.findall(line)
        if clean_decimals:
            continue
        digits = "".join(ch for ch in line if ch.isdigit())
        if 3 <= len(digits) <= 4:
            first_value = float(f"{digits[0]}.{digits[1:3]}")
            break

    if first_value is None:
        raise RuntimeError("Failed to parse Luzon first-month LWAP.")

    second_candidates: List[float] = []
    for line in block:
        second_candidates.extend(float(x) for x in DECIMAL_RE.findall(line))
    if not second_candidates:
        raise RuntimeError("Failed to parse Luzon second-month LWAP.")

    return first_value, second_candidates[-1]


def _extract_visayas_values(lines: List[str], mindanao_slide_idx: int) -> tuple[float, float]:
    visayas_idx = next(i for i, line in enumerate(lines) if line == "VISAYAS")
    block = lines[visayas_idx:mindanao_slide_idx]
    nums: List[float] = []
    for line in block:
        nums.extend(float(x) for x in DECIMAL_RE.findall(line))

    if len(nums) < 7:
        raise RuntimeError("Failed to parse Visayas LWAP values.")
    return nums[0], nums[6]


def _extract_mindanao_values(lines: List[str]) -> tuple[float, float]:
    mindanao_idx = [i for i, line in enumerate(lines) if line == "MINDANAO"][-1]
    zamboanga_idx = next(i for i, line in enumerate(lines[mindanao_idx + 1 :], start=mindanao_idx + 1) if "ZAMBOANGA" in line)

    first_nums: List[float] = []
    for line in lines[mindanao_idx + 1 : zamboanga_idx]:
        first_nums.extend(float(x) for x in DECIMAL_RE.findall(line))
    if not first_nums:
        raise RuntimeError("Failed to parse Mindanao first-month LWAP.")
    first_value = first_nums[0]

    second_value = None
    for line in lines[zamboanga_idx + 1 :]:
        nums = [float(x) for x in DECIMAL_RE.findall(line)]
        if not nums:
            continue
        if len(nums) == 1:
            second_value = nums[0]
            break
        if len(nums) >= 4:
            second_value = nums[3]
            break
    if second_value is None:
        raise RuntimeError("Failed to parse Mindanao second-month LWAP.")

    return first_value, second_value


def parse_prices_pages(pdf_path: Path) -> List[PricesPageParse]:
    parsed_pages: List[PricesPageParse] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln and ln.strip()]
            if len(lines) < 4:
                continue

            first_month = month_label_to_timestamp(lines[0])
            second_month = month_label_to_timestamp(lines[2])

            visayas_slide_idx = next(i for i, line in enumerate(lines) if "VISAYAS SLIDE" in line)
            mindanao_slide_idx = next(i for i, line in enumerate(lines) if "MINDANAO SLIDE" in line)

            luzon_first, luzon_second = _extract_luzon_values(lines, visayas_slide_idx)
            visayas_first, visayas_second = _extract_visayas_values(lines, mindanao_slide_idx)
            mindanao_first, mindanao_second = _extract_mindanao_values(lines)

            parsed_pages.append(
                PricesPageParse(
                    page=page_idx,
                    first_month=first_month,
                    second_month=second_month,
                    first_values={
                        "Luzon": luzon_first,
                        "Visayas": visayas_first,
                        "Mindanao": mindanao_first,
                    },
                    second_values={
                        "Luzon": luzon_second,
                        "Visayas": visayas_second,
                        "Mindanao": mindanao_second,
                    },
                )
            )

    if not parsed_pages:
        raise RuntimeError(f"No prices pages parsed from {pdf_path}")
    return parsed_pages


def build_prices_overlap_rows(parsed_pages: Iterable[PricesPageParse]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for page in parsed_pages:
        for island in ISLANDS:
            rows.append(
                {
                    "Page": page.page,
                    "Billing_Month": page.first_month,
                    "Island": island,
                    "Window_Position": "First_Month",
                    "Value": page.first_values[island],
                }
            )
            rows.append(
                {
                    "Page": page.page,
                    "Billing_Month": page.second_month,
                    "Island": island,
                    "Window_Position": "Second_Month",
                    "Value": page.second_values[island],
                }
            )
    return pd.DataFrame(rows)


def aggregate_monthly_series(overlap_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if "Page" in overlap_df.columns:
        agg = (
            overlap_df.sort_values(group_cols + ["Page"])
            .groupby(group_cols, as_index=False)
            .agg(Value=("Value", "last"), Source_Page=("Page", "max"))
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
    else:
        agg = (
            overlap_df.groupby(group_cols, as_index=False)
            .agg(
                Value=("Value", "mean"),
            )
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
    return agg


def build_overlap_checks(overlap_df: pd.DataFrame, group_cols: List[str], check_name: str) -> pd.DataFrame:
    checks = (
        overlap_df.groupby(group_cols, as_index=False)
        .agg(Occurrences=("Value", "size"), Min_Value=("Value", "min"), Max_Value=("Value", "max"))
    )
    checks["Absolute_Difference"] = (checks["Max_Value"] - checks["Min_Value"]).abs()
    checks.insert(0, "Check", check_name)
    return checks


def build_market_checks(market_overlap: pd.DataFrame, market_monthly: pd.DataFrame) -> pd.DataFrame:
    overlap_checks = build_overlap_checks(
        overlap_df=market_overlap,
        group_cols=["Metric", "Billing_Month"],
        check_name="overlap_consistency",
    )

    pivot = market_monthly.pivot(index="Billing_Month", columns="Metric", values="Value").reset_index()
    pivot["Difference_GWh"] = (
        pivot["Metered_Quantity_GWh"] - (pivot["Bilateral_Quantity_GWh"] + pivot["Spot_Quantity_GWh"])
    )
    arithmetic_checks = pivot[["Billing_Month", "Difference_GWh"]].copy()
    arithmetic_checks.insert(0, "Check", "metered_equals_bilateral_plus_spot")
    arithmetic_checks["Metric"] = "Metered_Quantity_GWh"
    arithmetic_checks["Occurrences"] = 1
    arithmetic_checks["Min_Value"] = arithmetic_checks["Difference_GWh"]
    arithmetic_checks["Max_Value"] = arithmetic_checks["Difference_GWh"]
    arithmetic_checks["Absolute_Difference"] = arithmetic_checks["Difference_GWh"].abs()

    overlap_checks["Difference_GWh"] = float("nan")
    cols = [
        "Check",
        "Metric",
        "Billing_Month",
        "Occurrences",
        "Min_Value",
        "Max_Value",
        "Absolute_Difference",
        "Difference_GWh",
    ]
    return pd.concat([overlap_checks[cols], arithmetic_checks[cols]], ignore_index=True)


def build_prices_adjacent_checks(parsed_pages: List[PricesPageParse]) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []
    for i in range(len(parsed_pages) - 1):
        current = parsed_pages[i]
        nxt = parsed_pages[i + 1]
        for island in ISLANDS:
            current_second = current.second_values[island]
            next_first = nxt.first_values[island]
            checks.append(
                {
                    "Check": "adjacent_page_overlap",
                    "Island": island,
                    "From_Page": current.page,
                    "To_Page": nxt.page,
                    "Billing_Month": current.second_month,
                    "Current_Second_Value": current_second,
                    "Next_First_Value": next_first,
                    "Absolute_Difference": abs(current_second - next_first),
                }
            )
    return pd.DataFrame(checks)


def save_market_panel(market_monthly: pd.DataFrame, output_path: Path) -> None:
    pivot = market_monthly.pivot(index="Billing_Month", columns="Metric", values="Value").sort_index()
    x = pivot.index
    right_edge_x = mdates.date2num(pd.Timestamp(x.max()).to_pydatetime())

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, pivot["Metered_Quantity_GWh"], marker="o", linewidth=2.1, label="Metered Quantity")
    axes[0].plot(x, pivot["Bilateral_Quantity_GWh"], marker="o", linewidth=1.8, label="Bilateral Quantity")
    axes[0].plot(x, pivot["Spot_Quantity_GWh"], marker="o", linewidth=1.8, label="Spot Quantity")
    axes[0].set_title("IEMOP System-Wide Quantities", weight="bold")
    axes[0].set_ylabel("GWh")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].legend(
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(right_edge_x, 3000),
        bbox_transform=axes[0].transData,
    )

    axes[1].plot(x, pivot["ESSP_PHP_per_kWh"], marker="o", linewidth=2.0, color="#d1495b", label="ESSP")
    axes[1].set_ylabel("Effective Spot Settlement Price (PhP/kWh)", color="#d1495b")
    axes[1].tick_params(axis="y", labelcolor="#d1495b")
    axes[1].set_title("IEMOP Prices and Trading Amount", weight="bold")
    ax2 = axes[1].twinx()
    ax2.plot(
        x,
        pivot["Trading_Amount_Billion_PHP"],
        marker="s",
        linewidth=2.0,
        color="#2a9d8f",
        label="Trading Amount",
    )
    ax2.set_ylabel("Trading Amount (Billion PHP)", color="#2a9d8f")
    ax2.tick_params(axis="y", labelcolor="#2a9d8f")

    combined_handles = axes[1].get_lines() + ax2.get_lines()
    combined_labels = [line.get_label() for line in combined_handles]
    axes[1].legend(
        combined_handles,
        combined_labels,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(right_edge_x, 6.5),
        bbox_transform=axes[1].transData,
    )

    x_labels = [d.strftime("%b %Y") for d in x]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[1].set_xlabel("Billing Month")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_prices_panel(prices_monthly: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for island in ISLANDS:
        series = prices_monthly[prices_monthly["Island"] == island].sort_values("Billing_Month")
        ax.plot(series["Billing_Month"], series["Value"], marker="o", linewidth=2.0, label=island)

    ax.set_title("IEMOP LWAP by Island Grid", weight="bold")
    ax.set_ylabel("Load Weighted Average Price (PhP/kWh)")
    ax.set_xlabel("Billing Month")
    ax.legend(ncol=3, loc="upper left")
    ax.set_xticks(sorted(prices_monthly["Billing_Month"].unique()))
    ax.set_xticklabels([d.strftime("%b %Y") for d in sorted(prices_monthly["Billing_Month"].unique())], rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    output_csv = base_dir / "outputs" / "csv"
    output_png = base_dir / "outputs" / "png"
    output_csv.mkdir(parents=True, exist_ok=True)
    output_png.mkdir(parents=True, exist_ok=True)

    market_pdf = data_dir / "COMBINED_MARKET_TRANS.pdf"
    prices_pdf = data_dir / "COMBINED_PRICES.pdf"
    if not market_pdf.exists() or not prices_pdf.exists():
        raise FileNotFoundError("Expected both COMBINED_MARKET_TRANS.pdf and COMBINED_PRICES.pdf in iemop/data.")

    market_overlap = extract_market_rows(market_pdf)
    market_monthly = aggregate_monthly_series(market_overlap, ["Billing_Month", "Metric"])
    market_checks = build_market_checks(market_overlap, market_monthly)

    prices_pages = parse_prices_pages(prices_pdf)
    prices_overlap = build_prices_overlap_rows(prices_pages)
    prices_monthly = aggregate_monthly_series(prices_overlap, ["Billing_Month", "Island"])
    prices_overlap_checks = build_overlap_checks(
        overlap_df=prices_overlap,
        group_cols=["Island", "Billing_Month"],
        check_name="overlap_consistency",
    )
    prices_adjacent_checks = build_prices_adjacent_checks(prices_pages)
    prices_checks = pd.concat([prices_overlap_checks, prices_adjacent_checks], ignore_index=True, sort=False)

    market_csv = output_csv / "market_transactions_monthly.csv"
    market_checks_csv = output_csv / "market_transactions_validation_checks.csv"
    prices_csv = output_csv / "lwap_prices_monthly.csv"
    prices_checks_csv = output_csv / "lwap_prices_validation_checks.csv"
    market_panel = output_png / "market_transactions_verified_panel.png"
    prices_panel = output_png / "lwap_prices_verified_panel.png"

    market_monthly.sort_values(["Billing_Month", "Metric"]).to_csv(market_csv, index=False)
    market_checks.sort_values(["Check", "Metric", "Billing_Month"]).to_csv(market_checks_csv, index=False)
    prices_monthly.sort_values(["Billing_Month", "Island"]).to_csv(prices_csv, index=False)
    prices_checks.sort_values(["Check", "Island", "Billing_Month"]).to_csv(prices_checks_csv, index=False)

    save_market_panel(market_monthly, market_panel)
    save_prices_panel(prices_monthly, prices_panel)

    market_max_overlap = market_checks.loc[
        market_checks["Check"] == "overlap_consistency", "Absolute_Difference"
    ].max()
    market_max_balance_diff = market_checks.loc[
        market_checks["Check"] == "metered_equals_bilateral_plus_spot", "Absolute_Difference"
    ].max()
    prices_max_overlap = prices_checks.loc[
        prices_checks["Check"].isin(["overlap_consistency", "adjacent_page_overlap"]), "Absolute_Difference"
    ].max()

    print(f"Wrote: {market_csv}")
    print(f"Wrote: {market_checks_csv}")
    print(f"Wrote: {prices_csv}")
    print(f"Wrote: {prices_checks_csv}")
    print(f"Wrote: {market_panel}")
    print(f"Wrote: {prices_panel}")
    print(f"Market overlap max absolute difference: {market_max_overlap:.6f}")
    print(f"Market metered-vs-components max absolute difference (GWh): {market_max_balance_diff:.6f}")
    print(f"Prices overlap max absolute difference: {prices_max_overlap:.6f}")


if __name__ == "__main__":
    main()
