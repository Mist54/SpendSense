"""Clustering pipeline for SpendSense transaction descriptions.

This module implements an unsupervised learning pipeline that:
  1. Loads the selected transaction dataset.
  2. Uses only the ``description`` column (never ``category``) for model fitting.
  3. Converts descriptions into TF-IDF features.
  4. Groups descriptions into K-Means clusters.
  5. Saves the fitted pipeline and clustered results separately.

Key design decisions
--------------------
* **Dataset**: Uses ``Dataset/train.csv`` because it contains the full set of
  descriptions from the original dataset (minus ambiguous ones), and the
  supervised pipeline already uses it as the training source.
* **TF-IDF settings**: ``min_df=2``, ``max_df=0.95``, ``ngram_range=(1, 2)``,
  ``sublinear_tf=True`` — similar to the supervised pipeline but tuned for
  clustering (no dependence on category labels).
* **K-Means**: ``random_state=42`` for reproducibility.
* **Artifacts**: Saved under ``models/unsupervised/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
MODELS_DIR: Path = PROJECT_ROOT / "models" / "unsupervised"
INITIAL_TRAIN_CSV: Path = DATASET_DIR / "train.csv"


def get_model_paths(output_dir: Path) -> dict[str, Path]:
    """Return a dict of artifact paths relative to the output directory."""
    return {
        "model": output_dir / "transaction_cluster_model.joblib",
        "clustered": output_dir / "clustered_transactions.csv",
        "summary": output_dir / "cluster_summary.csv",
        "metadata": output_dir / "clustering_metadata.json",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dataset(path: Path = INITIAL_TRAIN_CSV) -> pd.DataFrame:
    """Load the dataset and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"description", "category"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Handle missing descriptions safely
    df = df.copy()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    empty_mask = df["description"] == ""
    if empty_mask.any():
        logger.warning("Dropping %d rows with missing/empty descriptions", empty_mask.sum())
        df = df[~empty_mask]

    # Deduplicate by description
    df = df.drop_duplicates(subset=["description"]).reset_index(drop=True)
    return df


def build_clustering_pipeline(k: int = 10, random_state: int = 42) -> Pipeline:
    """Create a TF-IDF + K-Means pipeline for clustering.

    The pipeline fits TF-IDF on the description column only and K-Means on the
    resulting vectors. No category labels are used.
    """
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return Pipeline(steps=[("tfidf", tfidf), ("kmeans", kmeans)])


def run_clustering(
    k: int = 10,
    random_state: int = 42,
    dataset_path: Path = INITIAL_TRAIN_CSV,
    output_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Run the full clustering pipeline and save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = get_model_paths(output_dir)

    df = load_dataset(dataset_path)
    descriptions = df["description"].tolist()

    pipeline = build_clustering_pipeline(k=k, random_state=random_state)
    cluster_labels = pipeline.fit_predict(descriptions)

    df = df.copy()
    df["cluster_id"] = cluster_labels

    # Save artifacts
    joblib.dump(pipeline, paths["model"])
    df.to_csv(paths["clustered"], index=False)

    metadata: dict[str, Any] = {
        "k": k,
        "random_state": random_state,
        "n_samples": int(len(df)),
        "n_vocabulary": len(pipeline.named_steps["tfidf"].vocabulary_),
        "inertia": float(pipeline.named_steps["kmeans"].inertia_),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Clustering complete: K=%d, %d transactions, %d vocabulary features",
                k, len(df), metadata["n_vocabulary"])
    return {"pipeline": pipeline, "dataframe": df, "metadata": metadata}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run K-Means clustering on transactions.")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)
    run_clustering(k=args.k, random_state=args.random_state)


if __name__ == "__main__":
    main()
