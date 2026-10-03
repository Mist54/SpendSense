"""Prepare leakage-safe train/test datasets for SpendSense (description -> category).

Read-only with respect to Dataset/transactions-synthetic.csv.
Creates:
    Dataset/ambiguous.csv   - descriptions mapping to more than one category
    Dataset/train.csv       - 80% of unique descriptions (stratified by category)
    Dataset/test.csv        - 20% of unique descriptions (stratified by category)

The split unit is the unique DESCRIPTION, so no description can appear in
both train and test. random_state=42 for reproducibility.

No ML: no vectorization, features, training, or evaluation happens here.
"""

from pathlib import Path

import pandas as pd

RANDOM_STATE = 42
TEST_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
ORIGINAL_CSV = DATASET_DIR / "transactions-synthetic.csv"

TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"
AMBIGUOUS_CSV = DATASET_DIR / "ambiguous.csv"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def md5_of(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    hash_before = md5_of(ORIGINAL_CSV)

    # ------------------------------------------------------------------
    # 1. Load and preserve original data
    # ------------------------------------------------------------------
    section("1. LOAD ORIGINAL DATA")
    df = pd.read_csv(ORIGINAL_CSV)
    original_rows = len(df)
    print(f"Original rows: {original_rows}")
    print(f"Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Identify ambiguous descriptions
    # ------------------------------------------------------------------
    section("2. AMBIGUOUS DESCRIPTIONS")
    cats_per_desc = df.groupby("description")["category"].agg(["nunique", "unique", "size"])
    amb = cats_per_desc[cats_per_desc["nunique"] > 1].copy()
    amb["categories"] = amb["unique"].map(lambda arr: " | ".join(sorted(arr)))
    ambiguous_out = amb.reset_index()[["description", "categories", "size"]].rename(
        columns={"size": "row_count"}
    )
    ambiguous_out.to_csv(AMBIGUOUS_CSV, index=False)
    ambiguous_descriptions = set(amb.index)
    n_ambiguous = len(ambiguous_descriptions)
    rows_excluded_ambiguity = int(df["description"].isin(ambiguous_descriptions).sum())
    print(f"Ambiguous descriptions: {n_ambiguous}")
    print(f"Rows in original affected by ambiguity: {rows_excluded_ambiguity}")
    print(ambiguous_out.to_string(index=False))
    print(f"Written: {AMBIGUOUS_CSV.name}")

    # ------------------------------------------------------------------
    # 3. Create modeling data (exclude ambiguous, then dedup pairs)
    # ------------------------------------------------------------------
    section("3. MODELING DATA")
    df_nonamb = df[~df["description"].isin(ambiguous_descriptions)]
    rows_after_dedup_all = int(df.drop_duplicates(subset=["description", "category"]).shape[0])
    rows_nonamb_before_dedup = len(df_nonamb)
    df_model = df_nonamb.drop_duplicates(subset=["description", "category"]).reset_index(drop=True)
    print(f"Rows after removing exact duplicate pairs (whole dataset): {rows_after_dedup_all}")
    print(f"Rows after excluding ambiguous descriptions: {rows_nonamb_before_dedup}")
    print(f"Final modeling rows (non-ambiguous, deduplicated): {len(df_model)}")

    # Sanity: every description now has exactly one category
    assert df_model.groupby("description")["category"].nunique().max() == 1

    # One row per unique description, with its single category
    desc_cat = df_model.drop_duplicates(subset=["description"])[["description", "category"]]
    unique_descriptions = len(desc_cat)
    print(f"Unique descriptions in modeling data: {unique_descriptions}")

    # ------------------------------------------------------------------
    # 4. Description-level stratified split (80/20, random_state=42)
    # ------------------------------------------------------------------
    section("4. DESCRIPTION-LEVEL STRATIFIED SPLIT")
    rng = pd.Series(desc_cat.index).sample(frac=1, random_state=RANDOM_STATE)
    desc_cat = desc_cat.loc[rng].reset_index(drop=True)

    train_idx, test_idx = [], []
    for category, group in desc_cat.groupby("category"):
        n = len(group)
        n_test = int(round(TEST_FRACTION * n))
        n_test = min(n_test, n - 1) if n > 1 else 0  # keep at least 1 in train
        g_test = group.head(n_test)
        g_train = group.tail(n - n_test)
        train_idx.append(g_train)
        test_idx.append(g_test)
        print(f"  {category:<15} total={n:<6} train={len(g_train):<6} test={len(g_test)}")

    train = pd.concat(train_idx).reset_index(drop=True)
    test = pd.concat(test_idx).reset_index(drop=True)

    # Deterministic final ordering
    train = train.sort_values("description").reset_index(drop=True)
    test = test.sort_values("description").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Write output files
    # ------------------------------------------------------------------
    section("5. WRITE OUTPUT FILES")
    train[["description", "category"]].to_csv(TRAIN_CSV, index=False)
    test[["description", "category"]].to_csv(TEST_CSV, index=False)
    print(f"Written: {TRAIN_CSV.name} ({len(train)} rows)")
    print(f"Written: {TEST_CSV.name} ({len(test)} rows)")

    # ------------------------------------------------------------------
    # 6. Validation checks
    # ------------------------------------------------------------------
    section("6A. ROW COUNTS")
    print(f"Original rows: {original_rows}")
    print(f"Rows after removing exact duplicate (description, category) pairs: {rows_after_dedup_all}")
    print(f"Rows after excluding ambiguous descriptions: {rows_nonamb_before_dedup}")
    print(f"Final modeling rows (deduplicated, non-ambiguous): {len(df_model)}")
    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")

    section("6B. CATEGORY DISTRIBUTION (train / test)")
    for name, part in (("TRAIN", train), ("TEST", test)):
        counts = part["category"].value_counts()
        dist = pd.DataFrame({
            "category": counts.index,
            "count": counts.values,
            "percentage": (counts.values / counts.sum() * 100).round(2),
        })
        print(f"\n{name}:")
        print(dist.to_string(index=False))

    section("6C. LEAKAGE CHECK")
    shared = set(train["description"]) & set(test["description"])
    print(f"Unique descriptions train: {train['description'].nunique()}")
    print(f"Unique descriptions test:  {test['description'].nunique()}")
    print(f"Shared descriptions: {len(shared)}")
    if shared:
        print("FAIL: leakage detected ->", list(shared)[:10])
        raise SystemExit(1)
    print("Result: PASS")

    section("6D. AMBIGUOUS DATA")
    print(f"Number of ambiguous descriptions: {n_ambiguous}")
    print(f"Rows excluded because of ambiguity: {rows_excluded_ambiguity}")

    section("6E. CATEGORY COVERAGE")
    model_cats = set(df_model["category"])
    train_cats = set(train["category"])
    test_cats = set(test["category"])
    print(f"Categories in modeling data: {len(model_cats)}")
    print(f"Missing from train: {sorted(model_cats - train_cats) or 'none'}")
    print(f"Missing from test:  {sorted(model_cats - test_cats) or 'none'}")
    if model_cats - train_cats or model_cats - test_cats:
        print("FAIL: category coverage incomplete")
        raise SystemExit(1)
    print("Result: PASS")

    # ------------------------------------------------------------------
    # Original file untouched?
    # ------------------------------------------------------------------
    section("ORIGINAL FILE INTEGRITY")
    hash_after = md5_of(ORIGINAL_CSV)
    print(f"MD5 before: {hash_before}")
    print(f"MD5 after:  {hash_after}")
    print("Original dataset untouched:", hash_before == hash_after)

    print("\nPreparation complete. No ML performed.")


if __name__ == "__main__":
    main()
