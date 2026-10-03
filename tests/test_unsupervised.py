"""Unit tests for the SpendSense unsupervised clustering pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import unittest

from src.unsupervised.clustering import (
    build_clustering_pipeline,
    load_dataset,
    run_clustering,
)
from src.unsupervised.inference import load_cluster_model, predict_cluster


class TestUnsupervisedClustering(unittest.TestCase):

    def setUp(self) -> None:
        """Create a small synthetic CSV for testing."""
        self.tmp_dir = Path("test_output")
        self.tmp_dir.mkdir(exist_ok=True)
        data = {
            "description": [
                "WAL-MART #1234",
                "NETFLIX SUBSCRIPTION",
                "UBER TRIP",
                "ELECTRICITY BILL",
                "DOCTOR VISIT",
                "GROCERY STORE",
                "RESTAURANT",
                "HOTEL BOOKING",
                "MORTGAGE PAYMENT",
                "UTILITY BILL",
                "",  # missing description
                "WAL-MART #1234",  # duplicate
            ],
            "category": [
                "Shopping",
                "Subscription",
                "Transportation",
                "Utilities",
                "Healthcare",
                "Groceries",
                "Restaurants",
                "Travel",
                "Mortgage",
                "Utilities",
                "Fees",
                "Shopping",
            ],
        }
        self.synthetic_path = self.tmp_dir / "transactions.csv"
        pd.DataFrame(data).to_csv(self.synthetic_path, index=False)

    def tearDown(self) -> None:
        """Clean up test outputs."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_dataset_handles_missing_and_duplicates(self) -> None:
        df = load_dataset(self.synthetic_path)
        self.assertIn("description", df.columns)
        self.assertIn("category", df.columns)
        self.assertTrue((df["description"] != "").all())
        self.assertEqual(df["description"].nunique(), len(df))

    def test_pipeline_can_be_fitted(self) -> None:
        df = load_dataset(self.synthetic_path)
        pipeline = build_clustering_pipeline(k=3)
        labels = pipeline.fit_predict(df["description"])
        self.assertEqual(len(labels), len(df))
        self.assertTrue(set(labels).issubset({0, 1, 2}))

    def test_run_clustering_produces_artifacts(self) -> None:
        output_dir = self.tmp_dir / "models" / "unsupervised"
        result = run_clustering(
            k=3,
            random_state=42,
            dataset_path=self.synthetic_path,
            output_dir=output_dir,
        )
        df = result["dataframe"]
        self.assertIn("cluster_id", df.columns)
        self.assertGreater(len(df), 0)

        # Check artifacts exist
        self.assertTrue((output_dir / "transaction_cluster_model.joblib").exists())
        self.assertTrue((output_dir / "clustered_transactions.csv").exists())
        self.assertTrue((output_dir / "clustering_metadata.json").exists())

    def test_saved_model_can_be_loaded_and_predicts(self) -> None:
        output_dir = self.tmp_dir / "models" / "unsupervised"
        run_clustering(k=3, random_state=42, dataset_path=self.synthetic_path, output_dir=output_dir)

        loaded_pipeline = joblib.load(output_dir / "transaction_cluster_model.joblib")
        df = load_dataset(self.synthetic_path)
        labels = loaded_pipeline.predict(df["description"])
        self.assertEqual(len(labels), len(df))

    def test_reproducibility_same_seed(self) -> None:
        df = load_dataset(self.synthetic_path)
        pipeline1 = build_clustering_pipeline(k=3, random_state=42)
        labels1 = pipeline1.fit_predict(df["description"])

        pipeline2 = build_clustering_pipeline(k=3, random_state=42)
        labels2 = pipeline2.fit_predict(df["description"])

        self.assertEqual(list(labels1), list(labels2))

    def test_original_dataset_not_modified(self) -> None:
        original = pd.read_csv(self.synthetic_path)
        run_clustering(k=3, random_state=42, dataset_path=self.synthetic_path, output_dir=self.tmp_dir / "models")
        after = pd.read_csv(self.synthetic_path)
        pd.testing.assert_frame_equal(original, after)

    def test_supervised_pipeline_files_untouched(self) -> None:
        # Simply check that supervised files still exist
        self.assertTrue(Path("src/train_model.py").exists())
        self.assertTrue(Path("src/predict.py").exists())
        self.assertTrue(Path("models/expense_category_model.joblib").exists())


    def test_inference_produces_cluster_details(self) -> None:
        output_dir = self.tmp_dir / "models" / "unsupervised"
        run_clustering(k=3, random_state=42, dataset_path=self.synthetic_path, output_dir=output_dir)

        # Import inference after training so paths line up with defaults
        # when using the default MODELS_DIR; for test we use explicit paths.
        from src.unsupervised.inference import predict_cluster
        pipeline = joblib.load(output_dir / "transaction_cluster_model.joblib")
        df = pd.read_csv(output_dir / "clustered_transactions.csv")
        result = predict_cluster("UBER TRIP", pipeline, clustered_csv=output_dir / "clustered_transactions.csv")
        self.assertIn("cluster_id", result)
        self.assertIn("count", result)
        self.assertIn("top_terms", result)
        self.assertIn("examples", result)

    def test_empty_description_rejected(self) -> None:
        from src.unsupervised.inference import predict_cluster
        pipeline = build_clustering_pipeline(k=3)
        with self.assertRaises(ValueError):
            predict_cluster("   ", pipeline)


if __name__ == "__main__":
    unittest.main()
