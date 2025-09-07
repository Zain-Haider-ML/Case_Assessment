# RFQ Similarity Pipeline (Tasks B.1–B.4)

This repository-style deliverable contains a small, reproducible pipeline to enrich RFQs with grade properties and compute similarity between RFQs.

## What it does

- **B.1 — Reference join & missing values**
  - Normalizes grade keys (case, suffixes, aliases like `ST12→DC01`).
  - Parses range-like strings in `reference_properties.tsv` into numeric **min/max/mid**.
  - Aggregates duplicate grades (min of mins, max of maxes, recompute mid).
  - Left-joins RFQs with the reference; keeps nulls and adds flags (`ref_found`, `missing_any_ref_property`).

- **B.2 — Feature engineering**
  - **Dimensions** (e.g., `thickness_mm`, `width_mm`, `length_mm`): treated as **intervals**.
    - If only a single value is present, set `min=max=value`.
    - Per-dimension similarity = **1D IoU** (intersection-over-union) of intervals.
    - Dimension similarity = average across available dimensions.
  - **Categoricals**: `coating`, `finish`, `form`, `surface_type` as **exact match** (1/0) per field, averaged over available fields.
  - **Grade properties**: all `*_mid` numeric fields created in B.1.
    - Drop very sparse features (default: keep those with **≥30%** coverage).
    - Impute missing with feature mean; **z-score** per feature; similarity = **cosine** between vectors.

- **B.3 — Similarity calculation**
  - Aggregate score = weighted average (defaults):
    - `0.4 * dimension_iou + 0.2 * categorical_match + 0.4 * grade_cosine`.
  - If a component is missing for a pair, weights are **re-normalized** over available components.
  - Excludes self and near-duplicates (≥ 0.999999), returns **Top-3** per RFQ.

- **B.4 — Pipeline & documentation**
  - This `run.py` is the end-to-end script.
  - Results are written to the output directory (defaults to `/mnt/data`).

## Files

- `run.py` – the pipeline script (B.1–B.3).
- `top3.csv` – results (B.3 output).
- `rfq_enriched.csv` – intermediate enriched table from B.1 (handy for QA).
- *(Optional)* `requirements.txt` – if you want to install dependencies explicitly (see below).

## How to run

From a Python 3.9+ environment with `pandas` and `numpy` installed:

```bash
python run.py   --rfq /mnt/data/rfq.csv   --reference /mnt/data/reference_properties.tsv   --outdir /mnt/data   --coverage 0.30   --w_dim 0.4 --w_cat 0.2 --w_grade 0.4
```

Flags:

- `--rfq` path to the RFQ CSV (default: `/mnt/data/rfq.csv`)
- `--reference` path to the reference TSV (default: `/mnt/data/reference_properties.tsv`)
- `--outdir` output directory (default: `/mnt/data`)
- `--coverage` minimum coverage for grade features (default: `0.30`)
- `--w_dim`, `--w_cat`, `--w_grade` weights for the similarity components

Outputs created in `--outdir`:

- `rfq_enriched.csv` – RFQs + parsed grade properties (`*_min/*_max/*_mid`) and flags
- `top3.csv` – columns `[rfq_id, match_id, similarity_score]`

## Notes / Assumptions

- Grade normalization removes suffixes after `'+'` (e.g., `DX51D+Z140 → DX51D`) and maps common aliases (`ST12→DC01`, etc.).
- Range parsing supports bounds (`≤`, `≥`, `min`, `max`) and hyphenated ranges (`a-b`), including German decimals (`,`).
- Categorical field names are expected to be `coating`, `finish`, `form`, `surface_type` if present; case-insensitive normalization is applied.
- Dimensions considered by default: `thickness_mm`, `width_mm`, `length_mm`. Extend this list in `run.py` if needed.
- For similarity, weights are re-normalized if any component is unavailable for a pair.

## Dependencies

- Python 3.9+
- pandas
- numpy

(Optionally, create and use a virtual environment, then `pip install pandas numpy`).

## Reproducibility

- The script uses deterministic operations; similarity ordering can change only if the input data changes or if you adjust weights/coverage thresholds.
- All key steps are documented in the code with comments for maintainability.
