"""Interactive inference for the SpendSense unsupervised clustering pipeline.

Loads the saved TF-IDF + K-Means pipeline and assigns a new transaction
description to an existing cluster. It also provides cluster context:
size, important terms, and representative examples.

Important: K-Means assigns an existing cluster to new input; it does NOT
create a new cluster for every description. A cluster ID is not a semantic
expense category.
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


def load_cluster_model() -> Any:
    """Load the fitted clustering pipeline, with a clear error if missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run the unsupervised clustering training first (option 5 in main.py)."
        )
    return joblib.load(MODEL_PATH)


def get_cluster_details(cluster_id: int, pipeline, clustered_csv: Path = CLUSTERED_CSV) -> dict[str, Any]:
    """Return cluster size, top terms, and representative descriptions."""
    if not clustered_csv.exists():
        raise FileNotFoundError(
            f"Clustered data not found: {clustered_csv}\n"
            "Run clustering training/analysis first."
        )

    df = pd.read_csv(clustered_csv)
    cluster_df = df[df["cluster_id"] == cluster_id]

    if cluster_df.empty:
        return {
            "cluster_id": cluster_id,
            "count": 0,
            "top_terms": [],
            "examples": [],
        }

    tfidf = pipeline.named_steps["tfidf"]
    kmeans = pipeline.named_steps["kmeans"]
    terms = np.array(tfidf.get_feature_names_out())
    center = kmeans.cluster_centers_[cluster_id]
    top_indices = np.argsort(center)[-10:][::-1]
    top_terms = [terms[idx] for idx in top_indices]

    examples = cluster_df["description"].head(5).tolist()

    return {
        "cluster_id": cluster_id,
        "count": int(len(cluster_df)),
        "top_terms": top_terms,
        "examples": examples,
    }


def predict_cluster(description: str, pipeline, clustered_csv: Path = CLUSTERED_CSV) -> dict[str, Any]:
    """Predict the cluster ID for a new description and return details."""
    description = description.strip()
    if not description:
        raise ValueError("Description cannot be empty.")

    # Transform using the SAME fitted TF-IDF vectorizer
    cluster_id = int(pipeline.predict([description])[0])

    # Check for all-zero TF-IDF vector (unfamiliar words)
    tfidf = pipeline.named_steps["tfidf"]
    vector = tfidf.transform([description])
    if vector.nnz == 0:
        logger.warning(
            "None of the words in this description were seen during training. "
            "The cluster assignment may be unreliable."
        )

    details = get_cluster_details(cluster_id, pipeline, clustered_csv)
    details["description"] = description

    return details


def run_interactive_inference() -> None:
    """Interactive mode: read a description and print its cluster."""
    try:
        pipeline = load_cluster_model()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("Unsupervised Cluster Inference")
    print("Type a transaction description to find its cluster.")
    print("Type 'quit' or 'exit' to return to the menu.\n")

    while True:
        try:
            text = input("Enter transaction description: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            print("Please enter a non-empty description.\n")
            continue

        try:
            result = predict_cluster(text, pipeline)
            print(f"\nPredicted cluster: {result['cluster_id']}")
            print(f"Cluster size:      {result['count']} transactions")
            print(f"Top terms:         {', '.join(result['top_terms'])}")
            print("Representative descriptions:")
            for i, ex in enumerate(result["examples"], 1):
                print(f"  {i}. {ex}")
            print("\nNote: This is NOT a predicted expense category.\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    run_interactive_inference()
