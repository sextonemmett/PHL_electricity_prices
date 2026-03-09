from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import PreparedGeometry, prep


COUNTRIES = [
    "Philippines",
    "Singapore",
    "Indonesia",
    "Malaysia",
    "Thailand",
    "Vietnam",
]

ISLAND_GROUPS = ["Luzon", "Visayas", "Mindanao"]

PHL_ADM1_GEOJSON_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/41af8f1/releaseData/gbOpen/PHL/ADM1/"
    "geoBoundaries-PHL-ADM1_simplified.geojson"
)

PHL_REGION_TO_ISLAND_GROUP = {
    "NCR": "Luzon",
    "CAR": "Luzon",
    "Ilocos Region": "Luzon",
    "Cagayan Valley": "Luzon",
    "Central Luzon": "Luzon",
    "Calabarzon": "Luzon",
    "Mimaropa": "Luzon",
    "Bicol Region": "Luzon",
    "Western Visayas": "Visayas",
    "Central Visayas": "Visayas",
    "Eastern Visayas": "Visayas",
    "Zamboanga Peninsula": "Mindanao",
    "Northern Mindanao": "Mindanao",
    "Davao Region": "Mindanao",
    "Soccsksargen": "Mindanao",
    "Caraga": "Mindanao",
    "ARMM": "Mindanao",
    "BARMM": "Mindanao",
}


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


def get_base_paths() -> tuple[Path, Path, Path]:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, data_dir, output_dir


def load_philippines_island_polygons(base_dir: Path) -> Dict[str, BaseGeometry]:
    cache_dir = base_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = cache_dir / "geoBoundaries-PHL-ADM1_simplified.geojson"

    if not geojson_path.exists():
        urllib.request.urlretrieve(PHL_ADM1_GEOJSON_URL, geojson_path)  # noqa: S310

    with geojson_path.open("r", encoding="utf-8") as f:
        geo = json.load(f)

    grouped_geometries: Dict[str, List[BaseGeometry]] = {k: [] for k in ISLAND_GROUPS}
    for feature in geo.get("features", []):
        region_name = str(feature.get("properties", {}).get("shapeName", "")).strip()
        island_group = PHL_REGION_TO_ISLAND_GROUP.get(region_name)
        if island_group is None:
            continue
        grouped_geometries[island_group].append(shape(feature["geometry"]))

    polygons: Dict[str, BaseGeometry] = {}
    for island_group, geoms in grouped_geometries.items():
        if not geoms:
            raise RuntimeError(f"No geometries found for island group '{island_group}' from ADM1 data.")
        polygons[island_group] = unary_union(geoms).buffer(0)

    return polygons


def find_single_data_file(data_dir: Path) -> Path:
    files = [p for p in data_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one data file in {data_dir}, found {len(files)}: {[f.name for f in files]}"
        )
    return files[0]


def load_data(data_file: Path) -> pd.DataFrame:
    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_file, sheet_name="Power facilities")
    elif data_file.suffix.lower() == ".csv":
        df = pd.read_csv(data_file)
    else:
        raise RuntimeError(f"Unsupported data file type: {data_file.suffix}")
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def prepare_operating_data(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Country/area",
        "Type",
        "Capacity (MW)",
        "Start year",
        "Status",
        "Owner(s)",
        "Owner(s) GEM Entity ID",
        "Latitude",
        "Longitude",
    ]
    validate_columns(df, required)

    out = df.copy()
    out["Status"] = out["Status"].astype(str).str.strip().str.lower()
    out = out[out["Status"] == "operating"].copy()
    out["Country/area"] = out["Country/area"].astype(str).str.strip()
    out["Type"] = out["Type"].astype(str).str.strip()
    out["Capacity (MW)"] = pd.to_numeric(out["Capacity (MW)"], errors="coerce")
    out["Start year"] = pd.to_numeric(out["Start year"], errors="coerce")
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["Owner(s)"] = out["Owner(s)"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    out["Owner(s) GEM Entity ID"] = (
        out["Owner(s) GEM Entity ID"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )

    out = out[out["Capacity (MW)"].notna() & (out["Capacity (MW)"] > 0)].copy()
    current_year = date.today().year
    out["Age"] = np.where(
        out["Start year"].notna() & (out["Start year"] <= current_year),
        current_year - out["Start year"],
        np.nan,
    )
    return out


def build_type_color_map(types: List[str]) -> Dict[str, tuple]:
    palette = sns.color_palette("muted", n_colors=len(types))
    return dict(zip(types, palette))


def capacity_weighted_age(group: pd.DataFrame) -> float:
    valid = group[group["Age"].notna() & (group["Capacity (MW)"] > 0)]
    if valid.empty:
        return float("nan")
    return float(np.average(valid["Age"], weights=valid["Capacity (MW)"]))


def country_generation_mix(
    country_df: pd.DataFrame, countries: List[str], type_order: List[str], type_colors: Dict[str, tuple], output_dir: Path
) -> None:
    mix = (
        country_df.groupby(["Country/area", "Type"], as_index=False)["Capacity (MW)"]
        .sum()
        .rename(columns={"Capacity (MW)": "Capacity_MW"})
    )
    country_totals = mix.groupby("Country/area", as_index=False)["Capacity_MW"].sum().rename(
        columns={"Capacity_MW": "Country_Total_MW"}
    )
    mix = mix.merge(country_totals, on="Country/area", how="left")
    mix["Capacity_Share"] = mix["Capacity_MW"] / mix["Country_Total_MW"]

    pivot = (
        mix.pivot(index="Country/area", columns="Type", values="Capacity_Share")
        .reindex(index=countries, columns=type_order)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[type_colors[t] for t in pivot.columns],
        width=0.8,
        linewidth=0.2,
    )
    ax.set_title("Operating Capacity Mix by Country", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Capacity Share")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Generator Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "country_generation_mix_stacked.png", dpi=300)
    plt.close(fig)

    mix.to_csv(output_dir / "country_generation_mix_summary.csv", index=False)


def country_fleet_age(
    country_df: pd.DataFrame, countries: List[str], type_order: List[str], type_colors: Dict[str, tuple], output_dir: Path
) -> None:
    age_df = (
        country_df.groupby(["Country/area", "Type"], as_index=False)
        .apply(capacity_weighted_age, include_groups=False)
        .rename(columns={None: "Capacity_Weighted_Age_Years"})
    )
    age_df = age_df.dropna(subset=["Capacity_Weighted_Age_Years"])
    age_df["Country/area"] = pd.Categorical(age_df["Country/area"], categories=countries, ordered=True)
    age_df["Type"] = pd.Categorical(age_df["Type"], categories=type_order, ordered=True)
    age_df = age_df.sort_values(["Country/area", "Type"])

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        data=age_df,
        x="Country/area",
        y="Capacity_Weighted_Age_Years",
        hue="Type",
        hue_order=type_order,
        palette=type_colors,
        ax=ax,
    )
    ax.set_title("Capacity-Weighted Fleet Age by Technology and Country", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Capacity-Weighted Average Age (years)")
    ax.grid(axis="x", visible=False)
    ax.legend(title="Generator Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "country_fleet_age_grouped.png", dpi=300)
    plt.close(fig)

    age_df.to_csv(output_dir / "country_fleet_age_summary.csv", index=False)


def country_total_weighted_age(country_df: pd.DataFrame, countries: List[str], output_dir: Path) -> None:
    age_df = (
        country_df.groupby("Country/area", as_index=False)
        .apply(capacity_weighted_age, include_groups=False)
        .rename(columns={None: "Capacity_Weighted_Age_Years"})
    )
    age_df["Country/area"] = pd.Categorical(age_df["Country/area"], categories=countries, ordered=True)
    age_df = age_df.sort_values("Country/area")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=age_df,
        x="Country/area",
        y="Capacity_Weighted_Age_Years",
        order=countries,
        color="#4f7cac",
        ax=ax,
    )
    ax.set_title("Capacity-Weighted Average Fleet Age by Country", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Capacity-Weighted Average Age (years)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output_dir / "country_total_weighted_age.png", dpi=300)
    plt.close(fig)

    age_df.to_csv(output_dir / "country_total_weighted_age_summary.csv", index=False)


def ownership_concentration(country_df: pd.DataFrame, countries: List[str], output_dir: Path, top_n: int = 7) -> None:
    own = (
        country_df.groupby(["Country/area", "Owner(s)"], as_index=False)["Capacity (MW)"]
        .sum()
        .rename(columns={"Capacity (MW)": "Capacity_MW"})
    )
    totals = own.groupby("Country/area", as_index=False)["Capacity_MW"].sum().rename(
        columns={"Capacity_MW": "Country_Total_MW"}
    )
    own = own.merge(totals, on="Country/area", how="left")
    own["Capacity_Share"] = own["Capacity_MW"] / own["Country_Total_MW"]

    fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharex=False)
    axes = axes.ravel()
    for i, country in enumerate(countries):
        ax = axes[i]
        country_own = own[own["Country/area"] == country].sort_values("Capacity_MW", ascending=False)
        top = country_own.head(top_n).copy()
        rest_capacity = country_own.iloc[top_n:]["Capacity_MW"].sum()
        if rest_capacity > 0:
            rest_row = pd.DataFrame(
                [
                    {
                        "Country/area": country,
                        "Owner(s)": "Rest",
                        "Capacity_MW": rest_capacity,
                        "Country_Total_MW": country_own["Country_Total_MW"].iloc[0],
                        "Capacity_Share": rest_capacity / country_own["Country_Total_MW"].iloc[0],
                    }
                ]
            )
            plot_df = pd.concat([top, rest_row], ignore_index=True)
        else:
            plot_df = top

        ax.pie(
            plot_df["Capacity_MW"],
            labels=plot_df["Owner(s)"],
            autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 9},
        )
        ax.set_title(country, weight="bold")
    fig.suptitle("Ownership Share by Country (Top 7 Owners + Rest)", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "ownership_concentration_top7_plus_rest_pies.png", dpi=300)
    plt.close(fig)

    own.to_csv(output_dir / "ownership_concentration_summary.csv", index=False)


def classify_island_group(
    lat: float,
    lon: float,
    polygons: Dict[str, BaseGeometry],
    prepared_polygons: Dict[str, PreparedGeometry],
    prepared_buffered_polygons: Dict[str, PreparedGeometry],
) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    point = Point(lon, lat)

    # Primary pass: exact polygon match including border points.
    for name, poly in prepared_polygons.items():
        if poly.covers(point):
            return name

    # Secondary pass: include near-shore coordinates and slight geocoding imprecision.
    for name, poly in prepared_buffered_polygons.items():
        if poly.covers(point):
            return name

    # Tertiary pass: assign to nearest island polygon if reasonably close.
    distances = {name: point.distance(poly) for name, poly in polygons.items()}
    nearest_name = min(distances, key=distances.get)
    if distances[nearest_name] <= 1.2:
        return nearest_name

    # Otherwise, keep unclassified points explicit for review.
    return "Unknown"


def philippines_island_analysis(
    ph_df: pd.DataFrame, type_order: List[str], type_colors: Dict[str, tuple], output_dir: Path, base_dir: Path
) -> None:
    polygons = load_philippines_island_polygons(base_dir)
    prepared_polygons = {name: prep(poly) for name, poly in polygons.items()}
    prepared_buffered_polygons = {name: prep(poly.buffer(0.55)) for name, poly in polygons.items()}

    ph = ph_df.copy()
    ph["Island_Group"] = ph.apply(
        lambda x: classify_island_group(
            x["Latitude"],
            x["Longitude"],
            polygons,
            prepared_polygons,
            prepared_buffered_polygons,
        ),
        axis=1,
    )
    ph = ph[ph["Island_Group"].isin(ISLAND_GROUPS)].copy()

    island_mix = (
        ph.groupby(["Island_Group", "Type"], as_index=False)["Capacity (MW)"]
        .sum()
        .rename(columns={"Capacity (MW)": "Capacity_MW"})
    )
    island_totals = island_mix.groupby("Island_Group", as_index=False)["Capacity_MW"].sum().rename(
        columns={"Capacity_MW": "Island_Total_MW"}
    )
    island_mix = island_mix.merge(island_totals, on="Island_Group", how="left")
    island_mix["Capacity_Share"] = island_mix["Capacity_MW"] / island_mix["Island_Total_MW"]

    island_totals["Island_Group"] = pd.Categorical(island_totals["Island_Group"], categories=ISLAND_GROUPS, ordered=True)
    island_totals = island_totals.sort_values("Island_Group")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=island_totals,
        x="Island_Group",
        y="Island_Total_MW",
        order=ISLAND_GROUPS,
        color="#3f8f63",
        ax=ax,
    )
    ax.set_title("Philippines Total Operating Capacity by Island Group", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Total Capacity (MW)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output_dir / "philippines_island_total_capacity_bar.png", dpi=300)
    plt.close(fig)

    pivot = (
        island_mix.pivot(index="Island_Group", columns="Type", values="Capacity_Share")
        .reindex(index=ISLAND_GROUPS, columns=type_order)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[type_colors[t] for t in pivot.columns],
        width=0.72,
        linewidth=0.2,
    )
    ax.set_title("Philippines Operating Capacity Mix by Island Group", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Capacity Share")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Generator Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "philippines_island_generation_mix_stacked.png", dpi=300)
    plt.close(fig)

    island_age = (
        ph.groupby("Island_Group", as_index=False)
        .apply(capacity_weighted_age, include_groups=False)
        .rename(columns={None: "Capacity_Weighted_Age_Years"})
        .sort_values("Island_Group")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=island_age,
        x="Island_Group",
        y="Capacity_Weighted_Age_Years",
        order=ISLAND_GROUPS,
        color="#5f8f7b",
        ax=ax,
    )
    ax.set_title("Philippines Capacity-Weighted Fleet Age by Island Group", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Capacity-Weighted Average Age (years)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output_dir / "philippines_island_weighted_age.png", dpi=300)
    plt.close(fig)

    island_mix.to_csv(output_dir / "philippines_island_mix_summary.csv", index=False)
    island_age.to_csv(output_dir / "philippines_island_age_summary.csv", index=False)


def main() -> None:
    setup_style()
    base_dir, data_dir, output_dir = get_base_paths()
    data_file = find_single_data_file(data_dir)
    raw = load_data(data_file)
    operating = prepare_operating_data(raw)

    country_df = operating[operating["Country/area"].isin(COUNTRIES)].copy()
    type_order = (
        country_df.groupby("Type", as_index=False)["Capacity (MW)"]
        .sum()
        .sort_values("Capacity (MW)", ascending=False)["Type"]
        .tolist()
    )
    type_colors = build_type_color_map(type_order)

    country_generation_mix(country_df, COUNTRIES, type_order, type_colors, output_dir)
    country_fleet_age(country_df, COUNTRIES, type_order, type_colors, output_dir)
    country_total_weighted_age(country_df, COUNTRIES, output_dir)
    ownership_concentration(country_df, COUNTRIES, output_dir, top_n=7)

    ph_df = operating[operating["Country/area"] == "Philippines"].copy()
    philippines_island_analysis(ph_df, type_order, type_colors, output_dir, base_dir)

    totals = country_df.groupby("Country/area", as_index=False)["Capacity (MW)"].sum()
    totals["Capacity (MW)"] = totals["Capacity (MW)"].round(2)
    totals.to_csv(output_dir / "country_total_operating_capacity_mw.csv", index=False)

    print(f"Analysis complete. Outputs saved to: {output_dir}")
    print(f"Data source: {data_file.name}")
    print("Countries:", ", ".join(COUNTRIES))


if __name__ == "__main__":
    main()
