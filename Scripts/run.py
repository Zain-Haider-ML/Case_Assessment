

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Helpers shared across steps
# -----------------------------

def normalize_col(col: str) -> str:
    c = str(col).strip()
    c = re.sub(r"\s*\(mm\)\s*", "_mm", c, flags=re.I)
    c = re.sub(r"\s*\(kg\)\s*", "_kg", c, flags=re.I)
    c = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_").lower()
    return c

def grade_normalize(g: Optional[str]) -> Optional[str]:
    """Normalize grade keys (case, spacing, suffixes, aliases)."""
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    s = str(g).strip().upper()
    if "+" in s:
        s = s.split("+", 1)[0].strip()  # drop coating/condition suffixes
    s = re.sub(r"[\s\-_/]+", "", s)
    alias_map = {
        "ST12": "DC01", "ST13": "DC03", "ST14": "DC04", "ST15": "DC05", "ST16": "DC06",
        "SPCC": "DC01", "SPCD": "DC03", "SPCE": "DC04",
    }
    return alias_map.get(s, s)

def parse_range_to_min_max_mid(x) -> Tuple[float, float, float]:
    """Parse range-like strings to (min,max,mid) with support for ≤/≥/min/max/ranges and comma decimals."""
    if pd.isna(x):
        return (np.nan, np.nan, np.nan)
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x); return (v, v, v)
    s = str(x).lower()
    s = s.replace("%", " ").replace("wt.", " ").replace("mass", " ").replace("approx", " ")
    s = s.replace("–", "-").replace("—", "-").replace("to", "-").replace(",", ".")
    s = re.sub(r"\s+", " ", s).strip()

    min_tag = re.search(r"\bmin\s*([\-+]?\d*\.?\d+)", s)
    max_tag = re.search(r"\bmax\s*([\-+]?\d*\.?\d+)", s)
    if min_tag or max_tag:
        mn = float(min_tag.group(1)) if min_tag else np.nan
        mx = float(max_tag.group(1)) if max_tag else np.nan
        mid = np.nan if (np.isnan(mn) or np.isnan(mx)) else (mn + mx) / 2.0
        return (mn, mx, mid)

    le_match = re.search(r"(<=|=<|≤|<)\s*([\-+]?\d*\.?\d+)", s)
    ge_match = re.search(r"(>=|=>|≥|>)\s*([\-+]?\d*\.?\d+)", s)
    if le_match and not ge_match:
        mx = float(le_match.group(2)); return (np.nan, mx, np.nan)
    if ge_match and not le_match:
        mn = float(ge_match.group(2)); return (mn, np.nan, np.nan)

    range_match = re.match(r"^\s*([\-+]?\d*\.?\d+)\s*-\s*([\-+]?\d*\.?\d+)\s*$", s)
    if range_match:
        mn = float(range_match.group(1)); mx = float(range_match.group(2))
        return (mn, mx, (mn + mx) / 2.0)

    nums = re.findall(r"[\-+]?\d*\.?\d+", s)
    nums = [n for n in nums if n not in ["-", "+", ""]]
    if len(nums) >= 2:
        mn = float(nums[0]); mx = float(nums[1]); return (mn, mx, (mn + mx) / 2.0)
    if len(nums) == 1:
        v = float(nums[0]); return (v, v, v)
    return (np.nan, np.nan, np.nan)

def find_grade_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if re.search(r"grade", col, re.I):
            return col
    for col in df.columns:
        if re.search(r"material", col, re.I):
            return col
    return df.columns[0]

def find_id_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        if re.search(r"rfq.*id|id.*rfq", c, re.I):
            return c
    for c in cols:
        if c.lower() == "id":
            return c
    for c in cols:
        if c.lower().endswith("_id"):
            return c
    df["rfq_row_id"] = np.arange(len(df))
    return "rfq_row_id"


# -----------------------------
# B.1 — Reference join
# -----------------------------
def task_b1_join(rfq_path: Path, ref_path: Path, out_enriched_path: Path) -> pd.DataFrame:
    rfq = pd.read_csv(rfq_path)
    reference = pd.read_csv(ref_path, sep="\t")

    rfq.columns = [normalize_col(c) for c in rfq.columns]
    reference.columns = [normalize_col(c) for c in reference.columns]

    rfq_grade_col = find_grade_column(rfq)
    ref_grade_col = find_grade_column(reference)

    rfq["grade_norm"] = rfq[rfq_grade_col].apply(grade_normalize)
    reference["grade_norm"] = reference[ref_grade_col].apply(grade_normalize)
    reference = reference[reference["grade_norm"].notna()].copy()

    exclude_cols = {"grade_norm", rfq_grade_col, ref_grade_col}
    ref_obj_cols = [c for c in reference.columns if c not in exclude_cols and reference[c].dtype == "object"]
    ref_num_cols = [c for c in reference.columns if c not in exclude_cols and np.issubdtype(reference[c].dtype, np.number)]

    parsed_cols = []

    for col in ref_obj_cols:
        sample = reference[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        share_with_digit = sample.str.contains(r"\d").mean()
        if share_with_digit < 0.20:
            continue
        mins, maxs, mids = [], [], []
        for val in reference[col]:
            mn, mx, md = parse_range_to_min_max_mid(val)
            mins.append(mn); maxs.append(mx); mids.append(md)
        reference[f"{col}_min"] = mins
        reference[f"{col}_max"] = maxs
        reference[f"{col}_mid"] = mids
        parsed_cols.append(col)

    for col in ref_num_cols:
        reference[f"{col}_min"] = reference[col].astype(float)
        reference[f"{col}_max"] = reference[col].astype(float)
        reference[f"{col}_mid"] = reference[col].astype(float)
        parsed_cols.append(col)

    parsed_cols = sorted(set(parsed_cols))

    agg_dict = {}
    for col in parsed_cols:
        agg_dict[f"{col}_min"] = "min"
        agg_dict[f"{col}_max"] = "max"
    categoricals = [c for c in reference.columns
                    if c not in exclude_cols
                    and not c.endswith(("_min", "_max", "_mid"))
                    and c not in parsed_cols]
    for col in categoricals:
        agg_dict[col] = "first"

    ref_agg = reference.groupby("grade_norm", as_index=False).agg(agg_dict)

    for col in parsed_cols:
        mn_col, mx_col, mid_col = f"{col}_min", f"{col}_max", f"{col}_mid"
        ref_agg[mid_col] = np.where(
            ref_agg[mn_col].notna() & ref_agg[mx_col].notna(),
            (ref_agg[mn_col] + ref_agg[mx_col]) / 2.0,
            np.nan
        )

    rfq_enriched = rfq.merge(ref_agg, how="left", on="grade_norm", suffixes=("", "_ref"))
    rfq_enriched["ref_found"] = rfq_enriched["grade_norm"].isin(ref_agg["grade_norm"])

    parsed_min_cols = [f"{c}_min" for c in parsed_cols]
    parsed_max_cols = [f"{c}_max" for c in parsed_cols]
    parsed_mid_cols = [f"{c}_mid" for c in parsed_cols]
    parsed_all = [c for c in (parsed_min_cols + parsed_max_cols + parsed_mid_cols) if c in rfq_enriched.columns]

    def row_missing_any_ref(row):
        if not row["ref_found"]:
            return True
        if not parsed_all:
            return True
        return pd.isna(row[parsed_all]).all()

    rfq_enriched["missing_any_ref_property"] = rfq_enriched.apply(row_missing_any_ref, axis=1)

    rfq_enriched.to_csv(out_enriched_path, index=False)
    print(f"[B.1] Saved {out_enriched_path} | RFQs={len(rfq_enriched)} | matched={int(rfq_enriched['ref_found'].sum())}")
    return rfq_enriched


# -----------------------------
# B.2/B.3 — Feature engineering + similarity
# -----------------------------
def ensure_interval(df: pd.DataFrame, base: str) -> Tuple[Optional[str], Optional[str]]:
    mn = f"{base}_min" if f"{base}_min" in df.columns else None
    mx = f"{base}_max" if f"{base}_max" in df.columns else None
    if mn and mx:
        return mn, mx
    if base in df.columns:
        df[f"{base}_min"] = df[base]
        df[f"{base}_max"] = df[base]
        return f"{base}_min", f"{base}_max"
    return None, None

def interval_iou(a_min, a_max, b_min, b_max) -> float:
    if any(pd.isna([a_min, a_max, b_min, b_max])):
        return np.nan
    if a_max < a_min: a_min, a_max = a_max, a_min
    if b_max < b_min: b_min, b_max = b_max, b_min
    union_len = max(a_max, b_max) - min(a_min, b_min)
    inter_len = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    if union_len <= 0:
        return 1.0 if a_min == a_max == b_min == b_max else 0.0
    return inter_len / union_len

def compute_top3(
    rfq_enriched: pd.DataFrame,
    out_top3_path: Path,
    w_dim: float = 0.4,
    w_cat: float = 0.2,
    w_grade: float = 0.4,
    coverage_threshold: float = 0.30,
) -> pd.DataFrame:

    df = rfq_enriched.copy()
    rfq_id_col = find_id_col(df)

    # Categorical normalization
    for c in ["coating", "finish", "form", "surface_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    cat_cols = [c for c in ["coating", "finish", "form", "surface_type"] if c in df.columns]

    # Dimensions
    dim_pairs = {}
    for base in ["thickness_mm", "width_mm", "length_mm"]:
        mn, mx = ensure_interval(df, base)
        if mn and mx:
            dim_pairs[base] = (mn, mx)

    def pair_dimension_score(i: int, j: int) -> float:
        vals = []
        for _, (mn_col, mx_col) in dim_pairs.items():
            s = interval_iou(df.at[i, mn_col], df.at[i, mx_col],
                             df.at[j, mn_col], df.at[j, mx_col])
            if not pd.isna(s):
                vals.append(s)
        return float(np.mean(vals)) if vals else np.nan

    def pair_categorical_score(i: int, j: int) -> float:
        vals = []
        for c in cat_cols:
            a, b = df.at[i, c], df.at[j, c]
            if pd.isna(a) and pd.isna(b):
                continue
            vals.append(1.0 if (pd.notna(a) and pd.notna(b) and a == b) else 0.0)
        return float(np.mean(vals)) if vals else np.nan

    # Grade features
    mid_cols = [c for c in df.columns if c.endswith("_mid")]
    coverage = {c: 1 - df[c].isna().mean() for c in mid_cols}
    grade_feats = [c for c in mid_cols if coverage[c] >= coverage_threshold]

    if grade_feats:
        X = df[grade_feats].astype(float)
        col_means = X.mean(axis=0, skipna=True)
        X = X.fillna(col_means)
        col_stds = X.std(axis=0, ddof=0).replace(0, 1.0)
        X = (X - col_means) / col_stds
        G = X.to_numpy(dtype=float)
    else:
        G = np.zeros((len(df), 0))

    def pair_grade_score(i: int, j: int) -> float:
        if G.shape[1] == 0:
            return np.nan
        ai, bj = G[i], G[j]
        denom = (np.linalg.norm(ai) * np.linalg.norm(bj))
        if denom == 0:
            return np.nan
        return float(np.dot(ai, bj) / denom)

    # Aggregate similarity with renormalized weights
    def aggregate_similarity(i: int, j: int) -> float:
        s_dim = pair_dimension_score(i, j)
        s_cat = pair_categorical_score(i, j)
        s_grd = pair_grade_score(i, j)

        comps, weights = [], []
        if not pd.isna(s_dim): comps.append(s_dim); weights.append(w_dim)
        if not pd.isna(s_cat): comps.append(s_cat); weights.append(w_cat)
        if not pd.isna(s_grd): comps.append(s_grd); weights.append(w_grade)
        if not comps:
            return 0.0
        wsum = sum(weights)
        weights = [w/wsum for w in weights]
        return float(np.dot(comps, weights))

    n = len(df)
    rows = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            s = aggregate_similarity(i, j)
            if s >= 0.999999:
                continue
            sims.append((j, s))
        sims.sort(key=lambda t: t[1], reverse=True)
        top3 = sims[:3]
        rfq_id = df.at[i, rfq_id_col]
        for j, s in top3:
            match_id = df.at[j, rfq_id_col]
            rows.append((rfq_id, match_id, round(float(s), 6)))

    top3_df = pd.DataFrame(rows, columns=["rfq_id", "match_id", "similarity_score"])
    top3_df.to_csv(out_top3_path, index=False)

    print(f"[B.2+B.3] Saved {out_top3_path} | RFQs={n} | dims={list(dim_pairs.keys())} "
          f"| cats={cat_cols} | grade_feats={len(grade_feats)}")

    return top3_df


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="RFQ Similarity Pipeline (B.1–B.3)")
    parser.add_argument("--rfq", type=Path, default=Path("/mnt/data/rfq.csv"), help="Path to rfq.csv")
    parser.add_argument("--reference", type=Path, default=Path("/mnt/data/reference_properties.tsv"), help="Path to reference_properties.tsv")
    parser.add_argument("--outdir", type=Path, default=Path("/mnt/data"), help="Output directory")
    parser.add_argument("--coverage", type=float, default=0.30, help="Min coverage to keep a grade feature (default 0.30)")
    parser.add_argument("--w_dim", type=float, default=0.4, help="Weight for dimension IoU (default 0.4)")
    parser.add_argument("--w_cat", type=float, default=0.2, help="Weight for categorical match (default 0.2)")
    parser.add_argument("--w_grade", type=float, default=0.4, help="Weight for grade cosine (default 0.4)")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    enriched_path = args.outdir / "rfq_enriched.csv"
    top3_path = args.outdir / "top3.csv"

    # B.1
    rfq_enriched = task_b1_join(args.rfq, args.reference, enriched_path)

    # B.2+B.3
    _ = compute_top3(
        rfq_enriched=rfq_enriched,
        out_top3_path=top3_path,
        w_dim=args.w_dim,
        w_cat=args.w_cat,
        w_grade=args.w_grade,
        coverage_threshold=args.coverage,
    )

if __name__ == "__main__":
    main()




#python run.py --rfq .\resources\task_2\rfq.csv --reference .\resources\task_2\reference_properties.tsv --outdir .\outputs --coverage 0.30 --w_dim 0.4 --w_cat 0.2 --w_grade 0.4