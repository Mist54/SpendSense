"""Make category predictions with the saved SpendSense pipeline.

WHAT THIS SCRIPT DOES
---------------------
Loads the trained pipeline saved by train_model.py
(models/expense_category_model.joblib) and predicts an expense category for
any transaction description you give it, together with a confidence score.

It supports THREE input styles:
    1. Typed text       - single descriptions, typed or passed on the CLI
    2. A CSV file       - batch prediction for a whole file of descriptions
    3. Built-in samples - the five demo descriptions, used when nothing else
                          is provided

HOW THE SAVED PIPELINE PROCESSES RAW TEXT
-----------------------------------------
The saved object is a sklearn Pipeline with two fitted stages:
    1. TF-IDF vectorizer  - carries the vocabulary and IDF statistics LEARNED
       DURING TRAINING. Raw text is converted into the exact same kind of
       numerical vector the classifier saw while learning.
    2. Logistic Regression - the trained weight matrix that maps such a
       vector to 17 class probabilities.
Because both stages travel inside one object, we can feed RAW text - no
manual preprocessing, no recreating the vectorizer. This is exactly why we
saved the complete pipeline instead of only the classifier.

HOW predict() WORKS (CONCEPTUALLY)
----------------------------------
predict() runs the text through stage 1, then computes one score per class
in stage 2 and returns the class with the highest score.

WHAT predict_proba() REPRESENTS
-------------------------------
predict_proba() returns a probability for EVERY category (they sum to 1.0).
It is the model's internal confidence derived from its learned weights.

WHY CONFIDENCE IS NOT CERTAINTY
-------------------------------
These numbers are the model's OPINION, not a guarantee. They depend on the
training data (a perfectly balanced-feeling probability can still be wrong),
and Logistic Regression probabilities are known to be systematically
overconfident on text data. Treat high confidence as "the evidence pattern
strongly resembles training examples of that class", never as absolute
truth - and be extra skeptical when the top probabilities are close together.

USAGE
-----
    python src/predict.py "Uber ride to office"           # typed text
    python src/predict.py --csv my_transactions.csv       # batch from file
    python src/predict.py                                  # interactive prompt

CSV requirements: the file must contain a 'description' column (other
columns are allowed and ignored; the default 'category' column, if present,
is ignored too - predictions are compared against it only for information).

No category rules are hardcoded anywhere: every prediction comes from the
learned model.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "expense_category_model.joblib"

# The five sample descriptions used when no other input is given.
SAMPLE_DESCRIPTIONS = [
    "Uber ride to office",
    "Bought groceries from Walmart",
    "Netflix monthly subscription",
    "Electricity bill payment",
    "Hotel booking for vacation",
]


def load_pipeline():
    """Load the trained pipeline, with a clear error if it is missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run src/train_model.py first to train and save the model."
        )
    return joblib.load(MODEL_PATH)


def predict_with_confidence(pipeline, description: str) -> tuple[str, float, dict]:
    """Return (predicted_category, top_probability, full_probability_table).

    pipeline.predict([text])  -> the single most likely category.
    pipeline.predict_proba([text]) -> array of class probabilities in the
    order given by pipeline.classes_ (the model's known category list).
    We zip them together so every probability is attached to its class name.
    """
    predicted = pipeline.predict([description])[0]
    probabilities = pipeline.predict_proba([description])[0]
    probability_table = {
        class_name: round(float(prob), 4)
        for class_name, prob in zip(pipeline.classes_, probabilities)
    }
    top_probability = probability_table[predicted]
    return predicted, top_probability, probability_table


def print_prediction(description: str, predicted: str, confidence: float,
                     probability_table: dict) -> None:
    """Pretty-print one prediction, including the closest alternatives."""
    print(f"Input:       {description!r}")
    print(f"Predicted:   {predicted}  (confidence: {confidence:.2%})")
    # Show the three strongest alternatives to make the model's uncertainty
    # visible instead of hiding it behind a single number.
    top_three = sorted(probability_table.items(), key=lambda kv: kv[1], reverse=True)[:3]
    formatted = ", ".join(f"{name}: {prob:.2%}" for name, prob in top_three)
    print(f"Top 3:       {formatted}")


def predict_from_csv(pipeline, csv_path: Path) -> pd.DataFrame:
    """Batch-predict every description in a CSV file.

    WHY BATCH INSTEAD OF ONE-BY-ONE
    -------------------------------
    The pipeline can transform and score many rows in a single vectorized
    pass, which is far faster than calling predict() per row. We still
    print per-row results, but the model sees the whole batch at once.

    The CSV must contain a 'description' column. If it also has a real
    'category' column we keep it in the output and mark whether the model
    agreed - handy for eyeballing quality on labeled files, but this is NOT
    a formal evaluation (that is what test mode / evaluate_model() is for).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "description" not in df.columns:
        raise ValueError(
            f"{csv_path.name} must contain a 'description' column. "
            f"Found columns: {list(df.columns)}"
        )

    descriptions = df["description"].astype(str)
    predicted_categories = pipeline.predict(descriptions)
    all_probabilities = pipeline.predict_proba(descriptions)

    # predict_proba() rows are arrays of probabilities in the fixed order of
    # pipeline.classes_, so we build a class-name -> column-index lookup to
    # fetch each row's confidence for its own predicted class.
    class_index = {name: idx for idx, name in enumerate(pipeline.classes_)}

    result = pd.DataFrame({"description": descriptions})
    if "category" in df.columns:
        result["category_in_file"] = df["category"].values
    result["predicted_category"] = predicted_categories
    result["confidence"] = [
        round(float(probabilities[class_index[pred]]), 4)
        for probabilities, pred in zip(all_probabilities, predicted_categories)
    ]

    print(f"\nBatch prediction for {len(result)} rows from {csv_path.name}:")
    print("-" * 70)

    for _, row in result.iterrows():
        agreement = ""
        if "category_in_file" in result.columns:
            agreement = (
                "  [matches file]"
                if row["category_in_file"] == row["predicted_category"]
                else f"  [file says: {row['category_in_file']}]"
            )
        print(f"  {row['description'][:60]!r:<62} -> {row['predicted_category']:<15} ({row['confidence']:.2%}){agreement}")

    return result


def main(descriptions: list[str] | None = None,
         csv_path: str | None = None) -> None:
    """Predict for typed text and/or a CSV file.

    Priority: explicit arguments (passed by main.py's dispatcher) ->
    command-line arguments -> interactive prompt -> built-in samples.
    This keeps predict.py usable BOTH standalone and as a library called
    from the main.py dispatcher.
    """
    pipeline = load_pipeline()

    # --csv flag on the command line: everything after it may be a file path.
    if csv_path is None and "--csv" in (descriptions or sys.argv[1:]):
        tokens = descriptions if descriptions is not None else sys.argv[1:]
        flag_index = tokens.index("--csv")
        if flag_index + 1 < len(tokens):
            csv_path = tokens[flag_index + 1]

    if csv_path:
        result = predict_from_csv(pipeline, Path(csv_path))
        output_path = csv_path.rsplit(".", 1)[0] + "_predictions.csv" if isinstance(csv_path, str) else None
        if output_path:
            result.to_csv(output_path, index=False)
            print("-" * 70)
            print(f"Full results saved to: {output_path}")
        return

    # Typed-text mode.
    if descriptions is None:
        descriptions = sys.argv[1:]
    if not descriptions:
        try:
            typed = input("Enter a transaction description (Enter = run samples): ").strip()
        except EOFError:  # non-interactive run (e.g. automated test)
            typed = ""
        descriptions = [typed] if typed else SAMPLE_DESCRIPTIONS

    for description in descriptions:
        predicted, confidence, probability_table = predict_with_confidence(
            pipeline, description
        )
        print_prediction(description, predicted, confidence, probability_table)
        print()


if __name__ == "__main__":
    main()
