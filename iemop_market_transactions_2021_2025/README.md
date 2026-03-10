# IEMOP Market Transactions (2021-2025)

This folder scrapes IEMOP's yearly Market Operations Highlights pages for 2021 through 2025, downloads each report PDF, and extracts the `Market Transactions` slide into a monthly time series with overlap validation.

## Run

From repo root:

```bash
uv run python iemop_market_transactions_2021_2025/src/scrape_market_transactions.py
```

Generate QC plots:

```bash
uv run python iemop_market_transactions_2021_2025/src/make_qc_plots.py
```

## Outputs

CSV outputs are written to `iemop_market_transactions_2021_2025/outputs/csv/`:

- `reports_index_2021_2025.csv`: report metadata and download URLs scraped from IEMOP.
- `sample_reports_used_for_overlap.csv`: one representative report (December where available) per year and its billing-month span.
- `sample_variable_overlap_matrix.csv`: variable overlap matrix across sample PDFs (2021-2025).
- `market_transactions_raw_overlap_rows.csv`: all extracted rows (includes duplicate overlap observations from multiple reports).
- `market_transactions_monthly_latest.csv`: monthly series per metric and `market_scope`, keeping the latest available overlapping observation.
- `market_transactions_overlap_checks.csv`: overlap validation checks (`min`, `max`, `absolute_difference`) by metric, `market_scope`, and month.
- `parse_issues.csv`: reports that could not be parsed cleanly.

`market_scope` distinguishes historical definitions in the slide:
- `Luzon_Visayas`
- `Luzon_Visayas_Mindanao`

QC charts are written to `iemop_market_transactions_2021_2025/outputs/png/`:
- `qc_metric_small_multiples.png`
- `qc_energy_balance_by_scope.png`
- `qc_6line_quantities_lv_vs_lvm.png`
- `qc_essp_pre_post_2023_05.png`
- `qc_overlap_heatmap.png`
- `qc_top_overlap_discrepancies.png`
