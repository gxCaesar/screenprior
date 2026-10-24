# Saved-row AssayFormer comparison

This adapter reproduces the reported matched-pool comparison from saved scientific result tables. It performs no model inference, fitting, checkpoint download or raw-label reconstruction. Running it is not a reproduction of AssayFormer's checkpoint inference.

Install the release dependencies from the repository root, then run:

```bash
python -m pip install -e '.[dev]'
python comparison/report_assayformer.py --out comparison-report.json
```

The defaults read `data/study/assaybench20/screenprior_rows.csv` (200 ScreenPrior rows), `assayformer_matched_return_rows.csv` (200 native and 200 matched AssayFormer rows) and `screens.csv` (20 ordered screens and 15 source-defined publication groups). The separate `assayformer_native_rows.csv` preserves the first native return but is not concatenated again. The combined final-return file already contains its native cells.

To use copies elsewhere, pass `--screenprior-rows`, `--assayformer-rows` and `--metadata`. Omit `--out` to print JSON. An existing output file is refused.

The adapter requires seeds 0–9, 1,000 nominations, finite integer counts, exact phase/screen/seed coverage, consistent full-library counts and matching candidate-pool counts. It checks recorded enrichment factors against acquired hits, the original full-library hit rate and the effective in-library acquisition count. Missing seeds, duplicate cells, extra route rows, altered pools and changed metadata order or nested-subset membership are refused. These are checks of stored counts and denominators. Without raw libraries, hit masks and acquisition symbols, they do not independently verify gene-level pool identity or hit calls.

The estimand remains the equally weighted mean of 20 screen differences after averaging seeds within each screen. `ci95_screens` is the original 10,000-resample, seed-0 outcome interval. `ci95_publications` resamples whole publication groups and retains the screen-mean estimand. Publication-level averages do not replace that estimand. The `uncertainty.nested_never_read` interval is the later figure-derived interval for the nested 11-screen/seven-group subset. It is labelled separately and is not independent confirmation. The reported result is M3, with a screen-bootstrap interval crossing zero. That result establishes neither superiority nor equivalence. Native AssayFormer EF uses a different candidate universe and effective budget, so native values provide context rather than a fair performance comparison with ScreenPrior.

The signed-rank implementation explicitly chooses the exact distribution only for at most 50 non-zero differences without tied magnitudes. Otherwise it uses the tie-corrected normal approximation. All-zero differences have P=1. The original comparison's numerical functions and aggregation expressions are retained.

Run the focused checks from the repository root:

```bash
python -m pytest comparison/test_comparison.py -q
```

The tests use the released saved rows and deliberately corrupt copies in memory. They never fit or run an acquisition model. Source citations and data access conditions are described in the study-data documentation.
