from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

PHL_ADM1_GEOJSON_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/41af8f1/releaseData/gbOpen/PHL/ADM1/"
    "geoBoundaries-PHL-ADM1_simplified.geojson"
)

PALETTE = [
    "#C7252E",
    "#24557A",
    "#2E7D5B",
    "#96622D",
    "#6C4A9A",
    "#3D7C87",
    "#B55A3A",
    "#3E4A61",
    "#607D8B",
    "#4B6A4F",
]


def get_paths() -> tuple[Path, Path, Path, Path]:
    capacity_dir = Path(__file__).resolve().parents[2]
    data_dir = capacity_dir / "data"
    output_dir = capacity_dir / "map" / "outputs"
    cache_dir = capacity_dir / ".cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return capacity_dir, data_dir, output_dir, cache_dir


def find_single_data_file(data_dir: Path) -> Path:
    files = sorted([p for p in data_dir.iterdir() if p.is_file() and not p.name.startswith(".")])
    if len(files) == 1:
        return files[0]

    preferred = [p for p in files if "Global-Integrated-Power" in p.name]
    if len(preferred) == 1:
        return preferred[0]

    if not files:
        raise RuntimeError(f"No data files found in {data_dir}")

    raise RuntimeError(
        f"Expected one data file or one preferred GEM file in {data_dir}. Found: {[f.name for f in files]}"
    )


def load_raw_data(data_file: Path) -> pd.DataFrame:
    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_file, sheet_name="Power facilities")
    elif data_file.suffix.lower() == ".csv":
        df = pd.read_csv(data_file)
    else:
        raise RuntimeError(f"Unsupported data file type: {data_file.suffix}")

    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_philippines_generators(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Country/area",
        "Type",
        "Capacity (MW)",
        "Start year",
        "Status",
        "Owner(s)",
        "Operator(s)",
        "Latitude",
        "Longitude",
        "Plant / Project name",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    keep_cols = [
        "Plant / Project name",
        "Unit / Phase name",
        "Type",
        "Capacity (MW)",
        "Start year",
        "Status",
        "Owner(s)",
        "Operator(s)",
        "Latitude",
        "Longitude",
        "City",
        "Subnational unit (state, province)",
        "Major area (prefecture, district)",
        "GEM.Wiki URL",
        "Country/area",
    ]
    safe_keep_cols = [c for c in keep_cols if c in df.columns]
    ph = df[safe_keep_cols].copy()

    ph["Country/area"] = ph["Country/area"].astype(str).str.strip()
    ph["Status"] = ph["Status"].astype(str).str.strip().str.lower()
    ph["Type"] = ph["Type"].astype(str).str.strip()

    ph["Capacity (MW)"] = pd.to_numeric(ph["Capacity (MW)"], errors="coerce")
    ph["Start year"] = pd.to_numeric(ph["Start year"], errors="coerce")
    ph["Latitude"] = pd.to_numeric(ph["Latitude"], errors="coerce")
    ph["Longitude"] = pd.to_numeric(ph["Longitude"], errors="coerce")

    for col in [
        "Plant / Project name",
        "Unit / Phase name",
        "Owner(s)",
        "Operator(s)",
        "City",
        "Subnational unit (state, province)",
        "Major area (prefecture, district)",
        "GEM.Wiki URL",
    ]:
        if col in ph.columns:
            ph[col] = ph[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    ph = ph[
        (ph["Country/area"] == "Philippines")
        & (ph["Status"] == "operating")
        & ph["Capacity (MW)"].notna()
        & (ph["Capacity (MW)"] > 0)
        & ph["Latitude"].notna()
        & ph["Longitude"].notna()
    ].copy()

    if ph.empty:
        raise RuntimeError("No Philippines operating generators with valid coordinates and capacity were found.")

    ph = ph.sort_values("Capacity (MW)", ascending=False).reset_index(drop=True)
    return ph


def build_type_colors(ph: pd.DataFrame) -> Dict[str, str]:
    type_order = (
        ph.groupby("Type", as_index=False)["Capacity (MW)"]
        .sum()
        .sort_values("Capacity (MW)", ascending=False)["Type"]
        .tolist()
    )

    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(type_order)}


def build_marker_payload(ph: pd.DataFrame) -> List[dict]:
    records: List[dict] = []

    for _, row in ph.iterrows():
        start_year = None
        if pd.notna(row.get("Start year")):
            start_year = int(row["Start year"])

        records.append(
            {
                "plant_name": row.get("Plant / Project name", "Unknown"),
                "unit_name": row.get("Unit / Phase name", "Unknown"),
                "type": row["Type"],
                "capacity_mw": round(float(row["Capacity (MW)"]), 2),
                "start_year": start_year,
                "owner": row.get("Owner(s)", "Unknown"),
                "operator": row.get("Operator(s)", "Unknown"),
                "city": row.get("City", "Unknown"),
                "province": row.get("Subnational unit (state, province)", "Unknown"),
                "major_area": row.get("Major area (prefecture, district)", "Unknown"),
                "gem_url": row.get("GEM.Wiki URL", "Unknown"),
                "lat": round(float(row["Latitude"]), 6),
                "lon": round(float(row["Longitude"]), 6),
            }
        )

    return records


def build_size_legend_values(capacities: pd.Series) -> List[float]:
    quantiles = [0.25, 0.5, 0.9]
    values = [float(capacities.quantile(q)) for q in quantiles]
    values.append(float(capacities.max()))

    rounded = sorted({round(v, 1) for v in values if not math.isnan(v) and v > 0})
    if not rounded:
        rounded = [10.0, 100.0, 500.0]
    return rounded


def load_phl_adm1_geojson(cache_dir: Path) -> dict:
    geojson_path = cache_dir / "geoBoundaries-PHL-ADM1_simplified.geojson"
    if not geojson_path.exists():
        urllib.request.urlretrieve(PHL_ADM1_GEOJSON_URL, geojson_path)  # noqa: S310

    with geojson_path.open("r", encoding="utf-8") as f:
        geo = json.load(f)

    slim_features = []
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        slim_features.append(
            {
                "type": "Feature",
                "properties": {"shapeName": props.get("shapeName", "")},
                "geometry": feature.get("geometry"),
            }
        )

    return {"type": "FeatureCollection", "features": slim_features}


def build_html(
    points: List[dict],
    type_colors: Dict[str, str],
    size_legend_values: List[float],
    phl_adm1_geojson: dict,
) -> str:
    type_order = list(type_colors.keys())
    type_counts: Dict[str, int] = {t: 0 for t in type_order}
    for p in points:
        type_counts[p["type"]] = type_counts.get(p["type"], 0) + 1

    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Philippines Generator Map</title>
  <link
    rel=\"stylesheet\"
    href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"
    integrity=\"sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=\"
    crossorigin=\"\"
  />
  <style>
    :root {
      --paper: #f4efe5;
      --ink: #1e2a32;
      --sea: #dbe7ee;
      --accent: #c7252e;
      --land: #efe6d0;
      --panel: rgba(249, 246, 239, 0.93);
      --panel-border: rgba(89, 74, 56, 0.28);
    }

    html, body {
      margin: 0;
      height: 100%;
      background: var(--paper);
      color: var(--ink);
      font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, serif;
    }

    #map {
      width: 100%;
      height: 100%;
      background: var(--sea);
    }

    .title-card {
      position: absolute;
      top: 14px;
      right: 14px;
      z-index: 1000;
      max-width: 380px;
      padding: 10px 12px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      backdrop-filter: blur(2px);
      box-shadow: 0 8px 18px rgba(25, 37, 45, 0.15);
    }

    .title-card h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
      letter-spacing: 0.2px;
    }

    .title-card p {
      margin: 4px 0 0;
      font-size: 13px;
      line-height: 1.35;
      opacity: 0.85;
    }

    .economist-legend {
      font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, serif;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 10px 12px;
      box-shadow: 0 8px 18px rgba(25, 37, 45, 0.15);
      color: var(--ink);
      max-width: 280px;
    }

    .legend-title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 6px;
      border-bottom: 2px solid var(--accent);
      padding-bottom: 4px;
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0;
      font-size: 12.5px;
      line-height: 1.25;
    }

    .legend-toggle-row {
      cursor: pointer;
      user-select: none;
      transition: opacity 120ms ease-in-out;
    }

    .legend-toggle-row.is-off {
      opacity: 0.45;
    }

    .legend-toggle {
      margin: 0;
      width: 13px;
      height: 13px;
      accent-color: #2e4f6e;
      flex: 0 0 auto;
    }

    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(24, 30, 35, 0.55);
      flex: 0 0 auto;
    }

    .legend-label {
      display: inline-flex;
      align-items: baseline;
      gap: 5px;
    }

    .legend-count {
      opacity: 0.75;
      font-size: 11.5px;
    }

    .legend-divider {
      margin: 9px 0 7px;
      border-top: 1px solid rgba(30, 42, 50, 0.25);
    }

    .size-label {
      font-size: 12px;
      opacity: 0.9;
    }

    .plant-tooltip,
    .region-tooltip {
      border-radius: 5px;
      border: 1px solid rgba(40, 50, 58, 0.35);
      background: rgba(249, 246, 239, 0.96);
      color: var(--ink);
      box-shadow: 0 4px 10px rgba(17, 24, 28, 0.16);
      font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, serif;
    }

    .plant-tooltip b {
      color: #1a2832;
    }

    .leaflet-popup-content-wrapper,
    .leaflet-popup-tip {
      background: rgba(249, 246, 239, 0.98);
      color: var(--ink);
      border: 1px solid rgba(40, 50, 58, 0.35);
      box-shadow: 0 6px 14px rgba(17, 24, 28, 0.18);
    }

    .popup-title {
      font-size: 14px;
      margin: 0 0 5px;
      color: #1a2832;
    }

    .popup-meta {
      margin: 2px 0;
      font-size: 12.5px;
      line-height: 1.35;
    }

    .popup-link {
      margin-top: 6px;
      display: inline-block;
      color: #204e7d;
      text-decoration: none;
      border-bottom: 1px solid rgba(32, 78, 125, 0.45);
      font-size: 12.5px;
    }

    .popup-link:hover {
      color: #16395b;
      border-bottom-color: rgba(22, 57, 91, 0.75);
    }

    @media (max-width: 900px) {
      .title-card {
        max-width: 78vw;
      }

      .economist-legend {
        max-width: 66vw;
      }
    }
  </style>
</head>
<body>
  <div class=\"title-card\">
    <h1>Philippines Operating Generators</h1>
    <p>Circle size reflects capacity (MW). Color reflects generator type.</p>
  </div>
  <div id=\"map\"></div>

  <script
    src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"
    integrity=\"sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=\"
    crossorigin=\"\"
  ></script>
  <script>
    const points = __POINTS__;
    const typeColors = __TYPE_COLORS__;
    const typeOrder = __TYPE_ORDER__;
    const typeCounts = __TYPE_COUNTS__;
    const sizeLegendValues = __SIZE_VALUES__;
    const phlAdm1 = __PHL_ADM1__;

    const map = L.map("map", {
      minZoom: 5,
      maxZoom: 12,
      preferCanvas: true,
      zoomControl: true
    }).setView([12.65, 122.05], 6);

    map.createPane("boundaries");
    map.getPane("boundaries").style.zIndex = 340;
    map.createPane("markers");
    map.getPane("markers").style.zIndex = 420;

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: "abcd",
      maxZoom: 19
    }).addTo(map);

    L.geoJSON(phlAdm1, {
      pane: "boundaries",
      style: () => ({
        fillColor: "#efe6d0",
        fillOpacity: 0.23,
        color: "#c7252e",
        weight: 1.15,
        opacity: 0.66
      }),
      onEachFeature: (feature, layer) => {
        const region = feature.properties && feature.properties.shapeName ? feature.properties.shapeName : "Region";
        layer.bindTooltip(region, { sticky: true, direction: "center", className: "region-tooltip", opacity: 0.9 });
      }
    }).addTo(map);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: "abcd",
      maxZoom: 19,
      pane: "overlayPane"
    }).addTo(map);

    function safeText(value) {
      if (value === null || value === undefined || value === "") return "Unknown";
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    const capacities = points.map((p) => p.capacity_mw);
    const minCap = Math.min(...capacities);
    const maxCap = Math.max(...capacities);

    function radiusFromCapacity(capacityMw) {
      if (maxCap <= minCap) return 7;
      const transformed = (Math.sqrt(capacityMw) - Math.sqrt(minCap)) / (Math.sqrt(maxCap) - Math.sqrt(minCap));
      return 4 + transformed * 16;
    }

    const formatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

    function tooltipHtml(p) {
      const yearText = p.start_year === null ? "Unknown" : safeText(p.start_year);
      return `
        <div>
          <b>${safeText(p.plant_name)}</b><br>
          ${safeText(p.type)} | ${formatter.format(p.capacity_mw)} MW<br>
          Start year: ${yearText}<br>
          Owner: ${safeText(p.owner)}
        </div>
      `;
    }

    function popupHtml(p) {
      const yearText = p.start_year === null ? "Unknown" : safeText(p.start_year);
      const locationBits = [p.city, p.province, p.major_area].filter((x) => x && x !== "Unknown").map(safeText);
      const locationText = locationBits.length ? locationBits.join(", ") : "Unknown";
      const link = p.gem_url && p.gem_url !== "Unknown"
        ? `<a class=\"popup-link\" href=\"${safeText(p.gem_url)}\" target=\"_blank\" rel=\"noopener noreferrer\">GEM Wiki entry</a>`
        : "";

      return `
        <div>
          <h3 class=\"popup-title\">${safeText(p.plant_name)}</h3>
          <div class=\"popup-meta\"><b>Unit / phase:</b> ${safeText(p.unit_name)}</div>
          <div class=\"popup-meta\"><b>Type:</b> ${safeText(p.type)}</div>
          <div class=\"popup-meta\"><b>Capacity:</b> ${formatter.format(p.capacity_mw)} MW</div>
          <div class=\"popup-meta\"><b>Start year:</b> ${yearText}</div>
          <div class=\"popup-meta\"><b>Owner:</b> ${safeText(p.owner)}</div>
          <div class=\"popup-meta\"><b>Operator:</b> ${safeText(p.operator)}</div>
          <div class=\"popup-meta\"><b>Location:</b> ${locationText}</div>
          ${link}
        </div>
      `;
    }

    const groupsByType = {};
    typeOrder.forEach((t) => {
      groupsByType[t] = L.layerGroup().addTo(map);
    });

    points.forEach((p) => {
      const color = typeColors[p.type] || "#3f4c5a";
      const marker = L.circleMarker([p.lat, p.lon], {
        pane: "markers",
        radius: radiusFromCapacity(p.capacity_mw),
        color,
        weight: 1,
        opacity: 0.9,
        fillColor: color,
        fillOpacity: 0.72
      });

      marker.bindTooltip(tooltipHtml(p), {
        sticky: true,
        opacity: 0.94,
        className: "plant-tooltip",
        direction: "top"
      });
      marker.bindPopup(popupHtml(p), { maxWidth: 340 });

      if (!groupsByType[p.type]) {
        groupsByType[p.type] = L.layerGroup().addTo(map);
      }
      marker.addTo(groupsByType[p.type]);
    });

    const legendControl = L.control({ position: "bottomright" });
    legendControl.onAdd = () => {
      const div = L.DomUtil.create("div", "economist-legend");

      let typeRows = "";
      typeOrder.forEach((t, idx) => {
        const color = typeColors[t] || "#3f4c5a";
        typeRows += `
          <label class=\"legend-row legend-toggle-row\" data-index=\"${idx}\">
            <input class=\"legend-toggle\" type=\"checkbox\" data-index=\"${idx}\" checked />
            <span class=\"legend-dot\" style=\"background:${color}\"></span>
            <span class=\"legend-label\">${safeText(t)} <span class=\"legend-count\">(${typeCounts[t] || 0})</span></span>
          </label>
        `;
      });

      const sampleRows = sizeLegendValues.map((mw) => {
        const r = radiusFromCapacity(mw);
        const d = Math.max(7, r * 2);
        return `
          <div class=\"legend-row\">
            <span class=\"legend-dot\" style=\"width:${d}px;height:${d}px;background:rgba(34,54,73,0.2);border-color:rgba(34,54,73,0.6)\"></span>
            <span class=\"size-label\">${formatter.format(mw)} MW</span>
          </div>
        `;
      }).join("");

      div.innerHTML = `
        <div class=\"legend-title\">Generator Type (Toggle)</div>
        ${typeRows}
        <div class=\"legend-divider\"></div>
        <div class=\"legend-title\">Capacity Scale</div>
        ${sampleRows}
      `;

      const toggles = div.querySelectorAll(".legend-toggle");
      toggles.forEach((toggle) => {
        toggle.addEventListener("change", (event) => {
          const target = event.target;
          const typeIndex = Number(target.dataset.index);
          const typeName = typeOrder[typeIndex];
          if (!typeName || !groupsByType[typeName]) return;

          if (target.checked) {
            map.addLayer(groupsByType[typeName]);
          } else {
            map.removeLayer(groupsByType[typeName]);
          }

          const row = target.closest(".legend-toggle-row");
          if (row) {
            row.classList.toggle("is-off", !target.checked);
          }
        });
      });

      L.DomEvent.disableClickPropagation(div);
      return div;
    };

    legendControl.addTo(map);
  </script>
</body>
</html>
"""

    html = template.replace("__POINTS__", json.dumps(points, ensure_ascii=False))
    html = html.replace("__TYPE_COLORS__", json.dumps(type_colors, ensure_ascii=False))
    html = html.replace("__TYPE_ORDER__", json.dumps(type_order, ensure_ascii=False))
    html = html.replace("__TYPE_COUNTS__", json.dumps(type_counts, ensure_ascii=False))
    html = html.replace("__SIZE_VALUES__", json.dumps(size_legend_values, ensure_ascii=False))
    html = html.replace("__PHL_ADM1__", json.dumps(phl_adm1_geojson, ensure_ascii=False))
    return html


def main() -> None:
    _, data_dir, output_dir, cache_dir = get_paths()

    data_file = find_single_data_file(data_dir)
    raw = load_raw_data(data_file)
    ph = prepare_philippines_generators(raw)

    type_colors = build_type_colors(ph)
    points = build_marker_payload(ph)
    size_legend_values = build_size_legend_values(ph["Capacity (MW)"])
    phl_adm1_geojson = load_phl_adm1_geojson(cache_dir)

    html = build_html(points, type_colors, size_legend_values, phl_adm1_geojson)

    output_path = output_dir / "philippines_generator_map.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"Map generated: {output_path}")
    print(f"Source file: {data_file.name}")
    print(f"Generators plotted: {len(points)}")


if __name__ == "__main__":
    main()
