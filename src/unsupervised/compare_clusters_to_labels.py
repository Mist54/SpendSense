"""Optional post-clustering comparison with existing category labels.

This analysis compares discovered clusters with the true ``category`` column
AFTER training. It does not influence the clustering model, cluster
selection, or hyperparameter tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = PROJECT_ROOT / "models" / "unsupervised"
CLUSTERED_CSV: Path = MODELS_DIR / "clustered_transactions.csv"
COMPARISON_CSV: Path = MODELS_DIR / "cluster_category_comparison.csv"


def compare_clusters_to_categories() -> pd.DataFrame:
    """Compare cluster assignments with existing category labels."""
    if not CLUSTERED_CSV.exists():
        raise FileNotFoundError(
            "Clustered transactions not found. Run clustering.py first."
        )

    df = pd.read_csv(CLUSTERED_CSV)
    if "category" not in df.columns:
        logger.warning("Category column not present in clustered data; skipping comparison.")
        return pd.DataFrame()

    # Category counts per cluster
    counts = (
        df.groupby(["cluster_id", "category"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster_id", "count"], ascending=[True, False])
    )

    # Most common category per cluster
    most_common = (
        counts.sort_values(["cluster_id", "count"], ascending=[True, False])
        .drop_duplicates(subset=["cluster_id"])
        .rename(columns={"category": "most_common_category", "count": "most_common_count"})
    )

    # External metrics
    ari = adjusted_rand_score(df["category"], df["cluster_id"])
    nmi = normalized_mutual_info_score(df["category"], df["cluster_id"])

    comparison = most_common[["cluster_id", "most_common_category", "most_common_count"]].copy()
    comparison["ARI"] = round(ari, 4)
    comparison["NMI"] = round(nmi, 4)

    # Save
    comparison.to_csv(COMPARISON_CSV, index=False)
    logger.info("Saved cluster-category comparison to %s", COMPARISON_CSV)
    return comparison


if __name__ == "__main__":
    compare_clusters_to_categories()
