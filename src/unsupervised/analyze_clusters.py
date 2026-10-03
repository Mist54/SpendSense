"""Cluster analysis for SpendSense unsupervised pipeline.

Reads the clustered transaction output produced by ``clustering.py`` and
generates a human-readable summary of each cluster, including:

* Number of transactions
* Representative descriptions
* Top TF-IDF terms
* A short, content-based summary (no automatic semantic names)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = PROJECT_ROOT / "models" / "unsupervised"
MODEL_PATH: Path = MODELS_DIR / "transaction_cluster_model.joblib"
CLUSTERED_CSV: Path = MODELS_DIR / "clustered_transactions.csv"
SUMMARY_CSV: Path = MODELS_DIR / "cluster_summary.csv"


def _top_terms_per_cluster(pipeline, n_terms: int = 10) -> dict[int, list[str]]:
    """Return the top TF-IDF terms for each cluster centre."""
    tfidf = pipeline.named_steps["tfidf"]
    kmeans = pipeline.named_steps["kmeans"]
    terms = np.array(tfidf.get_feature_names_out())

    result: dict[int, list[str]] = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = np.argsort(center)[-n_terms:][::-1]
        result[i] = [terms[idx] for idx in top_indices]
    return result


def analyze_clusters(
    model_path: Path = MODEL_PATH,
    clustered_csv: Path = CLUSTERED_CSV,
    output_csv: Path = SUMMARY_CSV,
    n_terms: int = 10,
    max_examples: int = 5,
) -> pd.DataFrame:
    """Analyze discovered clusters and save a summary CSV."""
    if not model_path.exists() or not clustered_csv.exists():
        raise FileNotFoundError(
            "Clustering artifacts not found. Run clustering.py first to generate them."
        )

    pipeline = joblib.load(model_path)
    df = pd.read_csv(clustered_csv)

    top_terms = _top_terms_per_cluster(pipeline, n_terms=n_terms)

    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]
        examples = cluster_df["description"].head(max_examples).tolist()
        rows.append({
            "cluster_id": int(cluster_id),
            "count": int(len(cluster_df)),
            "top_terms": ", ".join(top_terms.get(int(cluster_id), [])),
            "examples": " | ".join(examples),
        })

    summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    summary.to_csv(output_csv, index=False)
    logger.info("Cluster analysis saved to %s (%d clusters)", output_csv, len(summary))
    return summary


if __name__ == "__main__":
    analyze_clusters()
