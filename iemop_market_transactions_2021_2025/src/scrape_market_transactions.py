from __future__ import annotations

import base64
import json
import re
import ssl
from dataclasses import dataclass
from pathlib import Path
import time
from urllib import error, parse, request

import pandas as pd
import pdfplumber


AJAX_URL = "https://www.iemop.ph/wp-admin/admin-ajax.php"
BASE_URL = "https://www.iemop.ph"
YEARS = [2021, 2022, 2023, 2024, 2025]
YEAR_PAGE_TEMPLATE = "https://www.iemop.ph/market-reports/{year}-market-operations-highlights/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

POST_ID_RE = re.compile(r'"post_id":"(\d+)"')
MONTH_YEAR_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    flags=re.IGNORECASE,
)
BILLING_TOKEN_RE = re.compile(r"(?:[A-Za-z]{3}-\d{2}|\d{2}-[A-Za-z]{3})")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
MONTH_ABBRS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_ABBR_TO_NUM = {abbr: idx + 1 for idx, abbr in enumerate(MONTH_ABBRS)}
MONTH_WORD_RE = re.compile(r"\b(" + "|".join(MONTH_ABBRS) + r")\b(?:\s*(\d{4}))?", flags=re.IGNORECASE)
MONTH_ROW_RE = re.compile(r"^\s*(" + "|".join(MONTH_ABBRS) + r")\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    label: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class ReportMeta:
    year_page: int
    report_id: int
    doc_title: str
    description: str
    published_date: str
    date_uploaded: str
    filename: str
    download_url: str
    report_month: pd.Timestamp | None


METRIC_SPECS = [
    MetricSpec(
        metric="Metered_Quantity_GWh",
        label="Metered Quantity (GWh)",
        patterns=(r"Metered Quantity(?:\s+\d{4})?\s*\(GWh\)",),
    ),
    MetricSpec(
        metric="Bilateral_Quantity_GWh",
        label="Bilateral Quantity (GWh)",
        patterns=(r"Bilateral Quantity(?:\s+\d{4})?\s*\(GWh\)",),
    ),
    MetricSpec(
        metric="Spot_Quantity_GWh",
        label="Spot Quantity (GWh)",
        patterns=(r"Spot Quantity(?:\s+\d{4})?\s*\(GWh\)",),
    ),
    MetricSpec(
        metric="Daily_Average_MQ_GWh",
        label="Daily Average MQ1 (GWh)",
        patterns=(r"Daily Average MQ1?\s*\(GWh\)|Daily Average MQ1?\(GWh\)",),
    ),
    MetricSpec(
        metric="Customer_ESSP_PHP_per_kWh",
        label="Customer ESSP2 (PhP/KWh)",
        patterns=(
            r"Customer ESSP2?\s*\(PhP/KWh\)|Customer ESSP2?\(PhP/KWh\)",
            r"^ESSP(?:\s+\d{4})?\s*\((?:P|PhP)/kWh\)",
        ),
    ),
    MetricSpec(
        metric="ESSP_PHP_per_kWh",
        label="ESSP2 (PhP/KWh)",
        patterns=(r"^ESSP2\s*\(PhP/KWh\)",),
    ),
    MetricSpec(
        metric="LuzVis_ESSP_PHP_per_kWh",
        label="Luz-Vis ESSP2 (PhP/KWh)",
        patterns=(r"Luz-Vis ESSP2\s*\(PhP/KWh\)",),
    ),
    MetricSpec(
        metric="Mindanao_ESSP_PHP_per_kWh",
        label="Mindanao ESSP2 (PhP/KWh)",
        patterns=(r"Mindanao ESSP2\s*\(PhP/KWh\)",),
    ),
    MetricSpec(
        metric="LVM_ESSP_PHP_per_kWh",
        label="LVM ESSP2 (PhP/KWh)",
        patterns=(r"LVM ESSP[23]\s*\(PhP/KWh\)",),
    ),
    MetricSpec(
        metric="Trading_Amount_Billion_PHP",
        label="Trading Amount (Billion PHP)",
        patterns=(r"^Trading Amount$",),
    ),
]

METRIC_PATTERNS = {
    spec.metric: [re.compile(pattern, flags=re.IGNORECASE) for pattern in spec.patterns]
    for spec in METRIC_SPECS
}


def http_get(url: str, data: bytes | None = None) -> bytes:
    retries = 4
    for attempt in range(1, retries + 1):
        try:
            req = request.Request(url=url, headers=HEADERS, data=data)
            with request.urlopen(req, timeout=60) as response:
                return response.read()
        except (error.URLError, TimeoutError, ssl.SSLError):
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("Unreachable retry loop in http_get.")


def sanitize_filename(value: str) -> str:
    return SAFE_RE.sub("_", value).strip("_")


def parse_report_month(text: str) -> pd.Timestamp | None:
    match = MONTH_YEAR_RE.search(text)
    if not match:
        return None
    month_name, year = match.groups()
    parsed = pd.to_datetime(f"{month_name.title()} {year}", format="%B %Y", errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_period("M").to_timestamp()


def month_token_to_timestamp(token: str) -> pd.Timestamp:
    for fmt in ("%b-%y", "%y-%b"):
        try:
            return pd.to_datetime(token, format=fmt).to_period("M").to_timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported month token: {token}")


def parse_numeric_tokens(text: str) -> list[float]:
    values: list[float] = []
    for token in NUMBER_RE.findall(text):
        cleaned = token.replace(",", "")
        if cleaned in {"", "-", "."}:
            continue
        values.append(float(cleaned))
    return values


def is_pdf_signature(path: Path) -> bool:
    signature = path.read_bytes()[:4]
    return signature.startswith(b"%PDF")


def discover_reports_for_year(year: int) -> list[ReportMeta]:
    page_url = YEAR_PAGE_TEMPLATE.format(year=year)
    html = http_get(page_url).decode("utf-8", errors="ignore")
    post_id_match = POST_ID_RE.search(html)
    if post_id_match is None:
        raise RuntimeError(f"Could not find post_id on page: {page_url}")

    payload = parse.urlencode(
        {
            "action": "display_filtered_market_reports_files",
            "sort": "",
            "datefilter": "",
            "page": 1,
            "post_id": post_id_match.group(1),
        }
    ).encode("utf-8")
    response = http_get(AJAX_URL, data=payload).decode("utf-8")
    parsed = json.loads(response)

    reports: list[ReportMeta] = []
    for source_id in parsed["source"]:
        row = parsed["data"][str(source_id)]
        doc_title = row.get("doc_title", "")
        filename = row.get("filename", "")
        title_blob = f"{doc_title} {filename}".lower()
        if "market operations highlights" not in title_blob:
            continue

        decoded_path = base64.b64decode(row["dl_path"]).decode("utf-8")
        public_path = decoded_path.replace("/var/www/html", "")
        download_url = BASE_URL + parse.quote(public_path, safe="/")

        report_month = parse_report_month(doc_title) or parse_report_month(filename)
        reports.append(
            ReportMeta(
                year_page=year,
                report_id=int(row["id"]),
                doc_title=doc_title,
                description=row.get("description", ""),
                published_date=row.get("published_date", ""),
                date_uploaded=row.get("date_uploaded", ""),
                filename=filename,
                download_url=download_url,
                report_month=report_month,
            )
        )

    return reports


def find_market_transactions_page(pdf_path: Path) -> tuple[int, list[str]]:
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if "MARKET TRANSACTIONS" not in text.upper():
                continue
            lines = [line.strip() for line in text.splitlines() if line and line.strip()]
            return idx, lines
    raise RuntimeError(f"Could not find 'Market Transactions' page in: {pdf_path}")


def _longest_contiguous_month_run(months: list[pd.Timestamp]) -> tuple[int, int]:
    if not months:
        return 0, 0
    ordinals = [month.to_period("M").ordinal for month in months]
    best_start = 0
    best_len = 1
    run_start = 0

    for idx in range(1, len(ordinals)):
        if ordinals[idx] == ordinals[idx - 1] + 1:
            continue
        run_len = idx - run_start
        if run_len > best_len:
            best_start = run_start
            best_len = run_len
        run_start = idx

    run_len = len(ordinals) - run_start
    if run_len > best_len:
        best_start = run_start
        best_len = run_len
    return best_start, best_len


def extract_billing_months(lines: list[str]) -> tuple[list[pd.Timestamp], int]:
    billing_idx = next((i for i, line in enumerate(lines) if "BILLING PERIOD" in line.upper()), None)
    if billing_idx is None:
        raise RuntimeError("Could not find 'BILLING PERIOD' line.")

    tokens: list[str] = []
    for offset in range(0, 5):
        idx = billing_idx + offset
        if idx >= len(lines):
            break
        line = lines[idx]
        if offset == 0:
            parts = re.split(r"BILLING PERIOD", line, flags=re.IGNORECASE, maxsplit=1)
            line = parts[1] if len(parts) == 2 else line
        tokens.extend(BILLING_TOKEN_RE.findall(line))

    if not tokens:
        raise RuntimeError("Could not parse billing month tokens.")

    parsed_months = [month_token_to_timestamp(token) for token in tokens]
    ordinals = [month.to_period("M").ordinal for month in parsed_months]
    is_contiguous = all(ordinals[idx] == ordinals[idx - 1] + 1 for idx in range(1, len(ordinals)))
    if is_contiguous:
        return parsed_months, 0

    best_start, best_len = _longest_contiguous_month_run(parsed_months)
    minimum_usable_run = max(6, len(parsed_months) // 2)
    if best_len >= minimum_usable_run:
        return parsed_months[best_start : best_start + best_len], best_start

    return parsed_months, 0


def infer_months_from_names(
    month_names: list[str],
    report_month: pd.Timestamp | None,
    assume_end_previous_month: bool = True,
) -> list[pd.Timestamp]:
    if not month_names:
        raise RuntimeError("No month names provided.")
    if report_month is None:
        raise RuntimeError("Cannot infer month years without report_month.")

    report_period = report_month.to_period("M")
    end_period = report_period - 1 if assume_end_previous_month else report_period
    start_period = end_period - (len(month_names) - 1)
    inferred = list(pd.period_range(start_period, end_period, freq="M").to_timestamp())
    inferred_names = [period.strftime("%b").upper() for period in pd.period_range(start_period, end_period, freq="M")]

    if inferred_names == [name.upper() for name in month_names]:
        return inferred

    alt_end_period = report_period
    alt_start_period = alt_end_period - (len(month_names) - 1)
    alt_names = [period.strftime("%b").upper() for period in pd.period_range(alt_start_period, alt_end_period, freq="M")]
    if alt_names == [name.upper() for name in month_names]:
        return list(pd.period_range(alt_start_period, alt_end_period, freq="M").to_timestamp())

    return inferred


def extract_settlement_months(lines: list[str], report_month: pd.Timestamp | None) -> list[pd.Timestamp]:
    idx = next((i for i, line in enumerate(lines) if "SETTLEMENT DATA" in line.upper()), None)
    if idx is None:
        raise RuntimeError("Could not find 'SETTLEMENT DATA' line.")

    source_text = " ".join(lines[idx : min(len(lines), idx + 3)])
    matches = MONTH_WORD_RE.findall(source_text)
    if not matches:
        raise RuntimeError("Could not parse month tokens from 'SETTLEMENT DATA'.")

    month_names = [month.upper()[:3] for month, _ in matches]
    years = [int(year) if year else None for _, year in matches]
    if all(year is None for year in years):
        return infer_months_from_names(month_names=month_names, report_month=report_month, assume_end_previous_month=True)

    resolved_months: list[pd.Timestamp] = []
    last_year: int | None = None
    for month_name, year in zip(month_names, years):
        if year is not None:
            last_year = year
        elif last_year is None and report_month is not None:
            last_year = report_month.year
        if last_year is None:
            raise RuntimeError("Could not resolve year for month token.")
        resolved_months.append(pd.Timestamp(year=last_year, month=MONTH_ABBR_TO_NUM[month_name], day=1))

    return resolved_months


def find_metric_match(lines: list[str], metric: str) -> tuple[int, re.Match[str]] | None:
    for line_idx, line in enumerate(lines):
        for regex in METRIC_PATTERNS[metric]:
            match = regex.search(line)
            if match:
                return line_idx, match
    return None


def is_other_metric_line(line: str, current_metric: str) -> bool:
    for metric, patterns in METRIC_PATTERNS.items():
        if metric == current_metric:
            continue
        if any(regex.search(line) for regex in patterns):
            return True
    return False


def align_partial_series(metric: str, values: list[float], months: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, float]]:
    if not values:
        return []

    month_count = len(months)
    if len(values) >= month_count:
        return list(zip(months, values[:month_count]))

    if metric == "LVM_ESSP_PHP_per_kWh":
        start = month_count - len(values)
    elif metric == "Mindanao_ESSP_PHP_per_kWh":
        start = 1 if (len(values) + 1) <= month_count else 0
    else:
        start = 0

    start = max(0, min(start, month_count - 1))
    end = min(month_count, start + len(values))
    trimmed = values[: max(0, end - start)]
    return list(zip(months[start:end], trimmed))


def extract_metric_values(
    lines: list[str],
    months: list[pd.Timestamp],
    metric: str,
    month_offset: int = 0,
) -> list[tuple[pd.Timestamp, float]]:
    metric_match = find_metric_match(lines, metric)
    if metric_match is None:
        return []

    line_idx, match = metric_match
    values = parse_numeric_tokens(lines[line_idx][match.end() :])

    for offset in range(1, 7):
        if len(values) >= len(months):
            break
        next_idx = line_idx + offset
        if next_idx >= len(lines):
            break
        next_line = lines[next_idx]
        if is_other_metric_line(next_line, current_metric=metric):
            break
        if any(char.isalpha() for char in next_line):
            break
        values.extend(parse_numeric_tokens(next_line))

    if month_offset > 0 and len(values) >= month_offset + len(months):
        values = values[month_offset:]

    return align_partial_series(metric=metric, values=values, months=months)


def detect_metric_presence(lines: list[str]) -> set[str]:
    present: set[str] = set()
    for spec in METRIC_SPECS:
        for line in lines:
            if any(regex.search(line) for regex in METRIC_PATTERNS[spec.metric]):
                present.add(spec.metric)
                break
    return present


def detect_market_scope(lines: list[str]) -> str:
    header_block = " ".join(lines[:12]).lower()
    if "luzon, visayas and mindanao" in header_block or "luzon visayas and mindanao" in header_block:
        return "Luzon_Visayas_Mindanao"
    if "luzon-visayas" in header_block or "luzon and visayas" in header_block:
        return "Luzon_Visayas"
    return "Unknown"


def extract_legacy_comparison_rows(
    lines: list[str],
    report_month: pd.Timestamp | None,
) -> dict[tuple[str, pd.Timestamp], float] | None:
    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "Metered Quantity (GWh)" in line and "Customer ESSP" in line and "Daily Average MQ" in line
        ),
        None,
    )
    if header_idx is None or report_month is None:
        return None

    metric_order = [
        "Metered_Quantity_GWh",
        "Bilateral_Quantity_GWh",
        "Spot_Quantity_GWh",
        "Daily_Average_MQ_GWh",
        "Customer_ESSP_PHP_per_kWh",
    ]
    year = report_month.year
    extracted: dict[tuple[str, pd.Timestamp], float] = {}

    for line in lines[header_idx + 1 :]:
        row_match = MONTH_ROW_RE.match(line)
        if row_match is None:
            continue
        month_abbr = row_match.group(1).upper()
        tail = line[row_match.end() :]
        entries = re.findall(r"-|\d[\d,]*\.?\d*", tail)
        if len(entries) < 15:
            continue

        billing_month = pd.Timestamp(year=year, month=MONTH_ABBR_TO_NUM[month_abbr], day=1)
        for idx, metric in enumerate(metric_order):
            token = entries[(idx * 3) + 2]
            if token == "-":
                continue
            extracted[(metric, billing_month)] = float(token.replace(",", ""))

    return extracted if extracted else None


def pick_sample_report(reports: list[ReportMeta], year: int) -> ReportMeta:
    year_reports = [report for report in reports if report.year_page == year]
    if not year_reports:
        raise RuntimeError(f"No reports found for year: {year}")

    december_reports = [
        report
        for report in year_reports
        if report.report_month is not None
        and report.report_month.year == year
        and report.report_month.month == 12
    ]
    if december_reports:
        return sorted(december_reports, key=lambda report: report.report_id)[-1]

    with_month = [report for report in year_reports if report.report_month is not None]
    if with_month:
        return sorted(with_month, key=lambda report: report.report_month)[-1]

    return sorted(year_reports, key=lambda report: report.report_id)[-1]


def save_reports_index(reports: list[ReportMeta], output_csv: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "year_page": report.year_page,
                "report_id": report.report_id,
                "doc_title": report.doc_title,
                "description": report.description,
                "published_date": report.published_date,
                "date_uploaded": report.date_uploaded,
                "filename": report.filename,
                "download_url": report.download_url,
                "report_month": report.report_month,
            }
            for report in reports
        ]
    )
    frame = frame.sort_values(["year_page", "report_month", "report_id"], na_position="last").reset_index(drop=True)
    frame["report_month"] = pd.to_datetime(frame["report_month"]).dt.strftime("%Y-%m-01")
    frame.to_csv(output_csv, index=False)
    return frame


def download_report(report: ReportMeta, target_dir: Path) -> Path:
    report_month_str = report.report_month.strftime("%Y_%m") if report.report_month is not None else "unknown_month"
    title_stub = sanitize_filename(report.doc_title or report.filename or str(report.report_id))
    filename = f"{report.year_page}_{report_month_str}_{report.report_id}_{title_stub}.pdf"
    target_path = target_dir / filename
    if target_path.exists():
        return target_path

    content = http_get(report.download_url)
    target_path.write_bytes(content)
    return target_path


def build_sample_overlap_matrix(sample_presence_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not sample_presence_rows:
        return pd.DataFrame(columns=["metric", "label"] + [str(year) for year in YEARS] + ["years_present"])

    long = pd.DataFrame(sample_presence_rows)
    matrix = (
        long.pivot_table(index=["metric", "label"], columns="year", values="present", aggfunc="max", fill_value=False)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for year in YEARS:
        col = str(year)
        if year in matrix.columns:
            matrix[col] = matrix[year].astype(bool)
            matrix = matrix.drop(columns=[year])
        else:
            matrix[col] = False
    matrix["years_present"] = matrix[[str(year) for year in YEARS]].sum(axis=1)
    return matrix.sort_values(["years_present", "metric"], ascending=[False, True]).reset_index(drop=True)


def build_overlap_checks(raw_rows: pd.DataFrame) -> pd.DataFrame:
    checks = (
        raw_rows.groupby(["metric", "market_scope", "billing_month"], as_index=False)
        .agg(
            occurrences=("value", "size"),
            source_reports=("report_id", "nunique"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .sort_values(["metric", "billing_month"])
        .reset_index(drop=True)
    )
    checks["absolute_difference"] = (checks["max_value"] - checks["min_value"]).abs()
    checks["billing_month"] = pd.to_datetime(checks["billing_month"]).dt.strftime("%Y-%m-01")
    return checks


def build_monthly_latest(raw_rows: pd.DataFrame) -> pd.DataFrame:
    raw_sorted = raw_rows.sort_values(["metric", "market_scope", "billing_month", "report_month", "report_id"])
    latest = (
        raw_sorted.groupby(["metric", "market_scope", "billing_month"], as_index=False)
        .tail(1)
        .sort_values(["metric", "market_scope", "billing_month"])
        .reset_index(drop=True)
    )

    coverage = (
        raw_rows.groupby(["metric", "market_scope", "billing_month"], as_index=False)
        .agg(source_reports=("report_id", "nunique"), overlap_rows=("value", "size"))
        .rename(columns={"source_reports": "source_report_count", "overlap_rows": "overlap_row_count"})
    )
    latest = latest.merge(coverage, on=["metric", "market_scope", "billing_month"], how="left")
    latest["billing_month"] = pd.to_datetime(latest["billing_month"]).dt.strftime("%Y-%m-01")
    latest["report_month"] = pd.to_datetime(latest["report_month"]).dt.strftime("%Y-%m-01")
    return latest


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "raw_pdfs"
    output_dir = base_dir / "outputs" / "csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[ReportMeta] = []
    for year in YEARS:
        all_reports.extend(discover_reports_for_year(year))

    reports_index_csv = output_dir / "reports_index_2021_2025.csv"
    save_reports_index(all_reports, reports_index_csv)

    raw_metric_rows: list[dict[str, object]] = []
    parse_issues: list[dict[str, object]] = []
    sample_presence_rows: list[dict[str, object]] = []
    sample_report_rows: list[dict[str, object]] = []

    sample_report_ids = {pick_sample_report(all_reports, year).report_id for year in YEARS}

    for report in sorted(all_reports, key=lambda item: (item.year_page, item.report_month, item.report_id)):
        local_pdf = download_report(report, data_dir)
        if not is_pdf_signature(local_pdf):
            parse_issues.append(
                {
                    "year_page": report.year_page,
                    "report_id": report.report_id,
                    "doc_title": report.doc_title,
                    "pdf_path": str(local_pdf),
                    "error": "Not a PDF file signature; likely Office file uploaded with .pdf extension.",
                }
            )
            continue

        try:
            page_number, lines = find_market_transactions_page(local_pdf)
            market_scope = detect_market_scope(lines)
            if market_scope == "Unknown" and report.year_page <= 2022:
                market_scope = "Luzon_Visayas"
        except Exception as exc:  # noqa: BLE001
            parse_issues.append(
                {
                    "year_page": report.year_page,
                    "report_id": report.report_id,
                    "doc_title": report.doc_title,
                    "pdf_path": str(local_pdf),
                    "error": str(exc),
                }
            )
            continue

        legacy_comparison_rows = extract_legacy_comparison_rows(lines=lines, report_month=report.report_month)
        billing_months: list[pd.Timestamp]
        month_offset = 0
        if legacy_comparison_rows is None:
            try:
                billing_months, month_offset = extract_billing_months(lines)
            except Exception:  # noqa: BLE001
                try:
                    billing_months = extract_settlement_months(lines=lines, report_month=report.report_month)
                    month_offset = 0
                except Exception as exc:  # noqa: BLE001
                    parse_issues.append(
                        {
                            "year_page": report.year_page,
                            "report_id": report.report_id,
                            "doc_title": report.doc_title,
                            "pdf_path": str(local_pdf),
                            "error": f"Could not parse billing months from BILLING PERIOD or SETTLEMENT DATA ({exc})",
                        }
                    )
                    continue
        else:
            billing_months = sorted({month for _, month in legacy_comparison_rows.keys()})

        if report.report_id in sample_report_ids:
            sample_report_rows.append(
                {
                    "year_page": report.year_page,
                    "report_id": report.report_id,
                    "doc_title": report.doc_title,
                    "report_month": report.report_month,
                    "pdf_path": str(local_pdf),
                    "market_transactions_page": page_number,
                    "billing_month_count": len(billing_months),
                    "billing_month_start": min(billing_months),
                    "billing_month_end": max(billing_months),
                    "market_scope": market_scope,
                }
            )

            present_metrics = detect_metric_presence(lines)
            for spec in METRIC_SPECS:
                sample_presence_rows.append(
                    {
                        "year": report.year_page,
                        "report_id": report.report_id,
                        "metric": spec.metric,
                        "label": spec.label,
                        "present": spec.metric in present_metrics,
                    }
                )

        report_level_rows: dict[tuple[str, pd.Timestamp], float] = {}
        if legacy_comparison_rows is not None:
            report_level_rows.update(legacy_comparison_rows)
        else:
            for spec in METRIC_SPECS:
                month_value_pairs = extract_metric_values(
                    lines=lines,
                    months=billing_months,
                    metric=spec.metric,
                    month_offset=month_offset,
                )
                for billing_month, value in month_value_pairs:
                    report_level_rows[(spec.metric, billing_month)] = value

        for (metric, billing_month), value in sorted(report_level_rows.items(), key=lambda item: (item[0][0], item[0][1])):
            raw_metric_rows.append(
                {
                    "year_page": report.year_page,
                    "report_id": report.report_id,
                    "doc_title": report.doc_title,
                    "report_month": report.report_month,
                    "download_url": report.download_url,
                    "pdf_path": str(local_pdf),
                    "market_transactions_page": page_number,
                    "market_scope": market_scope,
                    "billing_month": billing_month,
                    "metric": metric,
                    "value": value,
                }
            )

    if not raw_metric_rows:
        raise RuntimeError("No market transactions rows were extracted.")

    raw_rows = pd.DataFrame(raw_metric_rows).sort_values(
        ["metric", "billing_month", "report_month", "report_id"]
    ).reset_index(drop=True)
    raw_rows["billing_month"] = pd.to_datetime(raw_rows["billing_month"])
    raw_rows["report_month"] = pd.to_datetime(raw_rows["report_month"])

    raw_csv = output_dir / "market_transactions_raw_overlap_rows.csv"
    raw_out = raw_rows.copy()
    raw_out["billing_month"] = raw_out["billing_month"].dt.strftime("%Y-%m-01")
    raw_out["report_month"] = raw_out["report_month"].dt.strftime("%Y-%m-01")
    raw_out.to_csv(raw_csv, index=False)

    monthly_latest = build_monthly_latest(raw_rows)
    monthly_latest_csv = output_dir / "market_transactions_monthly_latest.csv"
    monthly_latest.to_csv(monthly_latest_csv, index=False)

    overlap_checks = build_overlap_checks(raw_rows)
    overlap_checks_csv = output_dir / "market_transactions_overlap_checks.csv"
    overlap_checks.to_csv(overlap_checks_csv, index=False)

    sample_overlap = build_sample_overlap_matrix(sample_presence_rows)
    sample_overlap_csv = output_dir / "sample_variable_overlap_matrix.csv"
    sample_overlap.to_csv(sample_overlap_csv, index=False)

    sample_reports_df = pd.DataFrame(sample_report_rows).sort_values("year_page").reset_index(drop=True)
    if not sample_reports_df.empty:
        sample_reports_df["report_month"] = pd.to_datetime(sample_reports_df["report_month"]).dt.strftime("%Y-%m-01")
        sample_reports_df["billing_month_start"] = pd.to_datetime(sample_reports_df["billing_month_start"]).dt.strftime(
            "%Y-%m-01"
        )
        sample_reports_df["billing_month_end"] = pd.to_datetime(sample_reports_df["billing_month_end"]).dt.strftime(
            "%Y-%m-01"
        )
    sample_reports_csv = output_dir / "sample_reports_used_for_overlap.csv"
    sample_reports_df.to_csv(sample_reports_csv, index=False)

    issues_csv = output_dir / "parse_issues.csv"
    pd.DataFrame(parse_issues).to_csv(issues_csv, index=False)

    max_overlap_diff = overlap_checks["absolute_difference"].max()
    non_zero_checks = overlap_checks[overlap_checks["absolute_difference"] > 0]

    print(f"Wrote {reports_index_csv}")
    print(f"Wrote {raw_csv}")
    print(f"Wrote {monthly_latest_csv}")
    print(f"Wrote {overlap_checks_csv}")
    print(f"Wrote {sample_overlap_csv}")
    print(f"Wrote {sample_reports_csv}")
    print(f"Wrote {issues_csv}")
    print(f"Extracted rows: {len(raw_rows)}")
    print(f"Distinct metrics: {raw_rows['metric'].nunique()}")
    print(f"Max overlap absolute difference: {max_overlap_diff:.6f}")
    print(f"Non-zero overlap checks: {len(non_zero_checks)}")
    print(f"Parse issues: {len(parse_issues)}")


if __name__ == "__main__":
    main()
