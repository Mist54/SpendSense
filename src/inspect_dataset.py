"""Inspect Dataset/transactions-synthetic.csv.

Read-only inspection. This script never writes to the dataset and performs
no splitting, cleaning, preprocessing, feature engineering, or training.
"""

from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "transactions-synthetic.csv"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    section("1. BASIC INFO")
    print(f"File: {DATASET_PATH.name}")
    print(f"Path: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names and dtypes:")
    print(df.dtypes.to_string())

    section("2. FIRST 10 ROWS")
    print(df.head(10).to_string())

    section("3. DATA QUALITY")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    n_duplicates = int(df.duplicated().sum())
    print(f"\nExact duplicate rows: {n_duplicates}")

    print("\nUnique values per column:")
    print(df.nunique(dropna=True).to_string())

    # Empty strings / whitespace-only, per object (text) column
    print("\nEmpty or whitespace-only values per object column:")
    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].astype("string").str.strip()
            n_empty = int(stripped.eq("").sum())
            print(f"  {col}: {n_empty}")

    section("4. OBVIOUS INVALID VALUES")
    for col in df.columns:
        print(f"\n[{col}] dtype={df[col].dtype}")
        if pd.api.types.is_numeric_dtype(df[col]):
            print(df[col].describe().to_string())
            negatives = df[df[col] < 0]
            print(f"negative values: {len(negatives)}")
            zeros = df[df[col] == 0]
            print(f"zero values: {len(zeros)}")
        else:
            vc = df[col].value_counts(dropna=False).head(10)
            print(f"top values (of {df[col].nunique(dropna=True)} unique):")
            print(vc.to_string())

    section("5. TARGET COLUMN IDENTIFICATION")
    print(f"Columns: {list(df.columns)}")

    desc_col = None
    cat_col = None

    # Candidate identification by exact/substring name match first
    name_map = {c.lower().strip(): c for c in df.columns}
    desc_aliases = ["description", "transaction_description", "desc", "details", "detail",
                    "narrative", "text", "memo", "transaction", "expense_description"]
    cat_aliases = ["category", "expense_category", "label", "class", "target",
                   "category_label", "type", "expense_type"]

    for alias in desc_aliases:
        if alias in name_map:
            desc_col = name_map[alias]
            break
    for alias in cat_aliases:
        if alias in name_map:
            cat_col = name_map[alias]
            break

    # Fallback: substring match
    if desc_col is None:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ("desc", "narrative", "memo", "detail", "text")):
                desc_col = c
                break
    if cat_col is None:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ("categ", "label", "class", "type")):
                cat_col = c
                break

    # Content-based fallback: object columns with many repeats -> label-like
    if cat_col is None:
        for c in df.columns:
            if df[c].dtype == object and 1 < df[c].nunique(dropna=True) <= 50:
                cat_col = c
                break

    if desc_col is None or cat_col is None:
        print("STOP: no suitable description and/or category column found.")
        return

    print(f"Proposed description column: {desc_col!r}")
    print(f"Proposed category column:    {cat_col!r}")

    section("6. CATEGORY DISTRIBUTION")
    counts = df[cat_col].value_counts(dropna=False)
    total = len(df)
    n_cat = counts.shape[0]
    print(f"Number of unique categories: {n_cat}")
    dist = pd.DataFrame({"count": counts, "percent": (counts / total * 100).round(2)})
    print(dist.to_string())
    imbalance = counts.iloc[0] / counts.iloc[-1]
    print(f"\nImbalance ratio (largest/smallest): {imbalance:.1f}x")
    if imbalance <= 3:
        verdict = "reasonably balanced"
    elif imbalance <= 10:
        verdict = "moderately imbalanced"
    else:
        verdict = "highly imbalanced"
    print(f"Verdict: classes are {verdict}.")

    section("7. TEXT STATISTICS (description column)")
    lengths = df[desc_col].astype("string").str.strip().str.len()
    print(f"Min description length:    {int(lengths.min())}")
    print(f"Max description length:    {int(lengths.max())}")
    print(f"Average description length: {lengths.mean():.2f}")
    print(f"Empty descriptions: {int(lengths.eq(0).sum())}")
    print("\n20 representative examples (one per row, first 20 rows):")
    for i, v in df[desc_col].head(20).items():
        print(f"  [{i}] {v!r}")

    section("8. DUPLICATE description+category COMBINATIONS")
    combo_dups = df.duplicated(subset=[desc_col, cat_col]).sum()
    unique_combos = df.groupby([desc_col, cat_col], dropna=False).ngroups
    print(f"Rows: {len(df)}")
    print(f"Unique (description, category) combinations: {unique_combos}")
    print(f"Extra rows beyond first occurrence: {int(combo_dups)}")
    if combo_dups:
        print("\nTop 10 duplicated combinations:")
        dup_counts = df.groupby([desc_col, cat_col]).size().sort_values(ascending=False)
        print(dup_counts.head(10).to_string())
        print("\nNote: the same description appearing with DIFFERENT categories:")
        multi = df.groupby(desc_col)[cat_col].nunique()
        conflicting = multi[multi > 1]
        print(f"{len(conflicting)} descriptions map to more than one category")

    print("\nInspection complete. Dataset was not modified.")


if __name__ == "__main__":
    main()
