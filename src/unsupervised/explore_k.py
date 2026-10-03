"""Explore different values of K for the SpendSense clustering pipeline.

Evaluates a range of K values using inertia (Elbow method) and silhouette
score on a sample of the data. Results are saved as JSON and a matplotlib
plot.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
MODELS_DIR: Path = PROJECT_ROOT / "models" / "unsupervised"
TRAIN_CSV: Path = DATASET_DIR / "train.csv"
EXPLORATION_JSON: Path = MODELS_DIR / "k_exploration.json"
ELBOW_PNG: Path = MODELS_DIR / "k_elbow_plot.png"


def explore_k(
    k_values: list[int] | None = None,
    sample_size: int = 5000,
    random_state: int = 42,
    dataset_path: Path = TRAIN_CSV,
) -> dict[str, Any]:
    """Evaluate inertia and silhouette for a range of K values."""
    if k_values is None:
        k_values = list(range(2, 21))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df = df[df["description"] != ""].drop_duplicates(subset=["description"])

    # Sample for evaluation to keep it lightweight
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        logger.info("Using %d sampled rows for K exploration", len(df))

    tfidf = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
    )
    X = tfidf.fit_transform(df["description"])

    results: dict[str, Any] = {"k_values": [], "inertia": [], "silhouette": []}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = float(kmeans.inertia_)
        sil = float(silhouette_score(X, labels, sample_size=min(sample_size, 5000), random_state=random_state))
        results["k_values"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(sil)
        logger.info("K=%d -> inertia=%.1f, silhouette=%.3f", k, inertia, sil)

    # Save JSON
    with open(EXPLORATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Save plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.plot(results["k_values"], results["inertia"], marker="o", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:orange")
    ax2.plot(results["k_values"], results["silhouette"], marker="s", color="tab:orange")
    ax1.set_title("Elbow and Silhouette for K Selection (sampled)")
    fig.tight_layout()
    fig.savefig(ELBOW_PNG, dpi=150)
    plt.close(fig)

    logger.info("K exploration results saved to %s and %s", EXPLORATION_JSON, ELBOW_PNG)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore K values for clustering.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    explore_k(sample_size=args.sample_size, random_state=args.random_state)
