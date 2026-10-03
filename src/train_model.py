"""Train the FIRST supervised-learning baseline model for SpendSense.
Logistic Regression --> Used Algorithm
WHAT THIS SCRIPT DOES
---------------------
It trains and evaluates a classic text-classification baseline that maps a
transaction description (e.g. "[debit] WAL-MART #0006") to an expense
category (e.g. "Shopping"), then saves the trained model for later reuse.

WHAT SUPERVISED LEARNING MEANS HERE
-----------------------------------
"Supervised" means we learn from EXAMPLES that already have the answer
attached: each row of our data is (description, correct category). The
algorithm's job is to discover the relationship between the words in a
description and its category, so it can guess the category for descriptions
it has never seen before.

INPUT  : Dataset/train.csv  (36,554 labeled descriptions - the "textbook")
         Dataset/test.csv   ( 9,138 labeled descriptions - the "final exam")
OUTPUT : models/expense_category_model.joblib  (reusable trained pipeline)
         models/confusion_matrix.png           (visual error map)
         models/misclassified.csv              (every test mistake, for study)

THE OVERALL ML PIPELINE (conceptual flow)
-----------------------------------------
    CSV  ->  raw text descriptions
         ->  TF-IDF numerical vectors        (text -> numbers)
         ->  Logistic Regression             (numbers -> category probabilities)
         ->  predicted category              (highest-probability class)
         ->  evaluation against true labels  (accuracy / precision / recall / F1)
         ->  error analysis + save pipeline

WHY THIS ORDER MATTERS
----------------------
Statistics (vocabulary, IDF weights) must be learned from TRAINING data only.
If test data influenced any preprocessing step, the "unseen data" evaluation
would be contaminated - that contamination is called DATA LEAKAGE, and it
makes reported accuracy dishonestly optimistic.

No neural networks, no embeddings, no hyperparameter tuning:
this is a deliberately simple, interpretable FIRST baseline.
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt  # noqa: F401  (explicit import required: `matplotlib` alone does not load pyplot)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths and configuration constants
# ---------------------------------------------------------------------------
# pathlib gives us OS-independent paths that work identically on Windows,
# macOS, and Linux. parents[1] from src/ is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "expense_category_model.joblib"
CONFUSION_PNG = MODELS_DIR / "confusion_matrix.png"
MISCLASSIFIED_CSV = MODELS_DIR / "misclassified.csv"

# random_state=42 "seeds" the random number generator. Logistic Regression
# has a couple of internally random steps (e.g. the order rows are processed
# by some solvers). Fixing the seed means re-running this script produces
# the exact same model - essential for reproducibility and for comparing
# future experiments fairly.
RANDOM_STATE = 42


def section(title: str) -> None:
    """Print a readable section header so script output is easy to follow."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and test.csv, verifying required columns exist.

    In this project:
      description = FEATURE  (the input information the model sees)
      category    = TARGET   (the answer we want the model to learn to predict)

    We use train.csv to teach the model and test.csv ONLY to grade it.
    The test file plays the role of "unseen real-world data" - if we let it
    influence anything during training, our evaluation would be a self-
    fulfilling prophecy instead of an honest estimate.
    """
    required_columns = {"description", "category"}

    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "Could not find train.csv / test.csv.\n"
            f"Expected at:\n  {TRAIN_CSV}\n  {TEST_CSV}\n"
            "Run src/prepare_dataset.py first to create the split files."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for name, df in (("train.csv", train_df), ("test.csv", test_df)):
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    print(f"Training samples:   {len(train_df)}")
    print(f"Testing samples:    {len(test_df)}")
    print(f"Categories (train): {train_df['category'].nunique()}")
    return train_df, test_df


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline.

    WHY A PIPELINE?
    ---------------
    A sklearn Pipeline chains preprocessing steps and the model into one
    object. Calling pipeline.predict(raw_text) then automatically runs
    TF-IDF followed by the classifier. This guarantees the exact same text
    processing used at training time is also applied at prediction time -
    a very common source of silent bugs when done manually - and lets us
    save/load the whole chain as ONE file.
    """
    vectorizer = TfidfVectorizer(
        # TF-IDF CONCEPT
        # --------------
        # Machine-learning models work with numbers, not words. TF-IDF
        # converts each text description into a numerical vector, one
        # dimension per vocabulary term.
        #
        #   TF  (term frequency): how often a term appears IN THIS document.
        #        A description mentioning "pmt" twice gets a higher TF for
        #        "pmt" than one mentioning it once.
        #
        #   IDF (inverse document frequency): how RARE a term is ACROSS ALL
        #        training documents = log(N / documents_containing_term).
        #        Terms in almost every description ("debit", "[") get a low
        #        IDF (little information); distinctive terms ("wal-mart")
        #        get a high IDF (highly informative).
        #
        #   TF-IDF score = TF x IDF  ->  "how important is this term for
        #   THIS description?" Frequent-in-this-text but rare-in-general
        #   terms score highest. Texts become sparse vectors of these scores.

        lowercase=True,
        # Convert all text to lowercase first, so "WAL-MART", "Wal-Mart" and
        # "wal-mart" are treated as the same word. Bank memos SHOUT, and we
        # do not want duplicate vocabulary for every capitalization variant.

        ngram_range=(1, 2),
        # A TOKEN is the smallest text unit (here: whitespace word).
        # A UNIGRAM is a single token ("walmart"); a BIGRAM is a pair of
        # consecutive tokens ("wal mart", "monthly subscription").
        # ngram_range=(1, 2) means "build features from unigrams AND
        # bigrams". Bigrams matter because some phrases are meaningful only
        # as a pair: "monthly interest" (income-ish) vs "interest" alone,
        # or "water bill" vs "water".

        min_df=2,
        # Minimum Document Frequency: ignore terms appearing in fewer than
        # 2 training descriptions. These are usually noise - account numbers,
        # one-off IDs, typos - that would bloat the vocabulary and could
        # cause overfitting to single training examples.

        max_df=0.95,
        # Maximum Document Frequency: ignore terms appearing in more than
        # 95% of descriptions. Such terms are near-universal boilerplate
        # (e.g. "debit") that barely help distinguish categories.

        sublinear_tf=True,
        # Replace raw term frequency with log-scaled frequency:
        #   1 + log(tf) instead of tf.
        # Rationale: a term appearing 10 times is NOT 10x more informative
        # than one appearing once - the relationship saturates. Log-scaling
        # dampens the influence of repeated words, which stabilizes learning.
    )

    classifier = LogisticRegression(
        # LOGISTIC REGRESSION CONCEPT
        # ---------------------------
        # Despite the name, it is a CLASSIFICATION algorithm. It computes a
        # weighted score for each class:
        #     score(class) = w1*x1 + w2*x2 + ... + b
        # where each x is a TF-IDF feature value and each w is a weight the
        # model LEARNS. A large positive weight on "uber" for the
        # Transportation class means "seeing 'uber' raises my belief that
        # this is Transportation". The score is squashed through the
        # logistic/sigmoid function into a probability between 0 and 1.
        #
        # MULTICLASS: with 17 categories, sklearn trains one
        # scores-vs-everything set of weights per class (one-vs-rest by
        # default for this solver) and predicts the class with the highest
        # probability. predict_proba() returns these 17 probabilities.
        #
        # WHY IT IS THE RIGHT FIRST BASELINE: fast on 36k samples, works
        # well on high-dimensional sparse text data, produces interpretable
        # weights, and gives a strong reference point. If a fancy model
        # cannot clearly beat it later, the fancy model is not worth its
        # complexity.

        max_iter=1000,
        # Maximum optimization iterations. The solver iteratively adjusts
        # weights until convergence; TF-IDF matrices are high-dimensional
        # and the default 100 iterations may stop early with a Convergence-
        # Warning (i.e. a half-trained model). 1000 gives it room to finish.

        random_state=RANDOM_STATE,
        # See RANDOM_STATE above: makes training reproducible run-to-run.
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def evaluate_model(pipeline: Pipeline, test_df: pd.DataFrame) -> dict:
    """Evaluate the trained pipeline on the UNSEEN test set.

    WHY test.csv MUST STAY UNSEEN
    -----------------------------
    During training the model fits its weights using train.csv only.
    test.csv is our stand-in for future real-world transactions. Comparing
    predictions against test labels estimates GENERALIZATION: how well the
    model performs on data it was never allowed to study. Measuring on the
    training data instead would mostly measure memorization (data leakage),
    producing inflated, useless scores.
    """
    section("EVALUATION ON TEST SET")

    X_test = test_df["description"]
    y_test = test_df["category"]

    y_pred = pipeline.predict(X_test)
    # Keep predictions as a pandas Series aligned with y_test so downstream
    # code (error analysis, plotting) can rely on consistent .values access.
    y_pred = pd.Series(y_pred, index=y_test.index)

    # ACCURACY: fraction of predictions that are exactly right.
    #   Misleading when classes are imbalanced - a model predicting the
    #   majority class for everything can score high while being useless.
    #   Our deduplicated data is moderately imbalanced (8.7% .. 2.9%), so we
    #   also report per-class and averaged metrics.
    accuracy = accuracy_score(y_test, y_pred)

    # PRECISION (per class): of all descriptions the model PREDICTED as class
    #   X, how many really were X? For us: "when it says 'Groceries', how
    #   often is it actually groceries?" Low precision = many false alarms.
    # RECALL (per class): of all descriptions that REALLY are class X, how
    #   many did the model FIND? For us: "of all real grocery transactions,
    #   what share did it catch?" Low recall = missed transactions.
    # F1: harmonic mean of precision and recall - high only when BOTH are
    #   high, so a single number punishes lopsided behavior.
    #
    # MACRO average: compute the metric per class, then take the simple mean.
    #   Every class counts EQUALLY - a mistake in tiny 'Fees' hurts macro as
    #   much as one in huge 'Shopping'. Sensitive to weak minority classes.
    # WEIGHTED average: per-class metrics averaged with weights = class
    #   frequency. Reflects the experience on a randomly drawn transaction.
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Weighted precision:  {precision_weighted:.4f}")
    print(f"Weighted recall:     {recall_weighted:.4f}")
    print(f"Weighted F1:         {f1_weighted:.4f}")
    print(f"Macro precision:     {precision_macro:.4f}")
    print(f"Macro recall:        {recall_macro:.4f}")
    print(f"Macro F1:            {f1_macro:.4f}")

    print("\nClassification report (per-category detail):")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    """Create and save a confusion-matrix heatmap.

    A CONFUSION MATRIX compares, for every test description, the model's
    prediction against the true category:
      - ROWS    = the ACTUAL (true) category
      - COLUMNS = the category the model PREDICTED
    The diagonal holds correct predictions; every off-diagonal cell is a
    specific mistake (which true class was mistaken for which other class).
    Reading the largest off-diagonal cells tells us exactly WHERE the model
    is confused - far more actionable than a single accuracy number.
    """
    section("CONFUSION MATRIX")

    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    figure, axis = plt.subplots(figsize=(11, 9))
    im = axis.imshow(matrix, cmap="Blues")

    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("Actual category")
    axis.set_title("SpendSense baseline - confusion matrix (test set)")

    # Annotate each cell with its count so the numbers are readable.
    threshold = matrix.max() / 2.0
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            axis.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    figure.colorbar(im, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(CONFUSION_PNG, dpi=150)
    plt.close(figure)
    print(f"Saved confusion matrix heatmap: {CONFUSION_PNG.name}")

    # Show the biggest confusion pairs in text form as well.
    pair_counts = (
        pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
        .query("actual != predicted")
        .value_counts(["actual", "predicted"])
        .rename("count")
        .reset_index()
    )
    print("\nMost common actual -> predicted confusions:")
    print(pair_counts.head(10).to_string(index=False))
    return pair_counts


def save_misclassified(test_df: pd.DataFrame, y_pred: pd.Series) -> pd.DataFrame:
    """Write every wrong prediction to models/misclassified.csv for study."""
    section("ERROR ANALYSIS")

    # Build a side-by-side table of truth vs. prediction, then keep the rows
    # where the model disagreed with reality. Aligning on .values avoids any
    # index-alignment surprises between test_df and the prediction Series.
    results = pd.DataFrame(
        {
            "description": test_df["description"].values,
            "actual_category": test_df["category"].values,
            "predicted_category": y_pred.values,
        }
    )
    wrong = results[results["actual_category"] != results["predicted_category"]]
    wrong.to_csv(MISCLASSIFIED_CSV, index=False)

    error_count = len(wrong)
    error_rate = error_count / len(test_df)
    print(f"Incorrect predictions: {error_count} of {len(test_df)}")
    print(f"Error rate:            {error_rate:.4f} ({error_rate:.2%})")
    print(f"Saved misclassified examples: {MISCLASSIFIED_CSV.name}")
    print(
        "\nWHY ERROR ANALYSIS MATTERS: accuracy is one number; it cannot tell "
        "you whether mistakes are random noise or a systematic pattern "
        "(e.g. 'everything pharmacy-ish gets called Shopping'). Reading real "
        "misclassified examples reveals fixable problems - label noise, "
        "missing features, genuinely ambiguous wording - and suggests what to "
        "improve NEXT, without blindly tuning."
    )
    return wrong


def load_training_data() -> pd.DataFrame:
    """Load ONLY train.csv and validate its columns.

    Used by train-and-save mode, which must never open test.csv - keeping
    the test set truly unseen is the whole point of that mode.
    """
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_CSV}.\n"
            "Run src/prepare_dataset.py first to create the split files."
        )
    train_df = pd.read_csv(TRAIN_CSV)
    missing = {"description", "category"} - set(train_df.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    return train_df


def load_test_data() -> pd.DataFrame:
    """Load ONLY test.csv and validate its columns (test mode helper)."""
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {TEST_CSV}.\n"
            "Run src/prepare_dataset.py first to create the split files."
        )
    test_df = pd.read_csv(TEST_CSV)
    missing = {"description", "category"} - set(test_df.columns)
    if missing:
        raise ValueError(f"test.csv is missing required columns: {sorted(missing)}")
    return test_df


def train_and_save_model(evaluate: bool = True) -> Pipeline:
    """Train the pipeline, save it, and optionally evaluate on the test set.

    WHY THE `evaluate` SWITCH EXISTS
    --------------------------------
    - evaluate=True  -> the full experiment: train, SAVE, then grade on the
      untouched test set (accuracy / precision / recall / F1, confusion
      matrix, error list).
    - evaluate=False -> 'train and save' only: fast when you already know
      the baseline quality and just want the model artifact. This mode never
      even OPENS test.csv, so no test information can influence anything.

    NOTE THE ORDER: the model is saved BEFORE evaluation runs, so even if
    the evaluation step somehow fails, the trained artifact is not lost.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Load the labeled data (train = textbook, test = final exam).
    train_df = load_training_data()
    test_df = load_test_data() if evaluate else None

    # 2/3. Separate features (X) from labels (y).
    #   X = the model's INPUT  (text descriptions)
    #   y = the model's TARGET (known correct categories)
    # We keep them apart because sklearn's API expects exactly this pairing:
    # fit(X, y) learns the mapping X -> y; predict(X) guesses y.
    X_train = train_df["description"]
    y_train = train_df["category"]

    # 4/5/6. Build pipeline and train.
    # "Training" = the optimizer adjusts the classifier's weights so that,
    # for the training vectors, the predicted category probabilities match
    # the known labels as closely as possible.
    #
    # LEAKAGE DETAIL: fit() here runs BOTH pipeline stages on TRAINING data
    # only - TF-IDF learns its vocabulary/IDF statistics from X_train, then
    # the classifier learns from the resulting vectors. test.csv (loaded
    # only when evaluate=True) touches nothing in this step.
    section("TRAINING (TF-IDF + Logistic Regression)")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    feature_count = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"Training complete. TF-IDF vocabulary size: {feature_count} features")

    # 10. Save the COMPLETE pipeline (vectorizer + classifier) as one file.
    # Saving the whole chain means a future script can do:
    #     pipeline = joblib.load(...)
    #     pipeline.predict(["[debit] WAL-MART #0006"])
    # with NO manual re-creation of preprocessing - the exact fitted TF-IDF
    # vocabulary and IDF weights travel inside the file. joblib is sklearn's
    # recommended serializer: it handles large numpy arrays efficiently.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline: {MODEL_PATH.name}")

    if evaluate:
        # 7/8. Honest evaluation on the untouched test set.
        metrics = evaluate_model(pipeline, test_df)
        plot_confusion_matrix(metrics["y_test"], metrics["y_pred"])

        # 9. Study the mistakes.
        misclassified = save_misclassified(test_df, metrics["y_pred"])
        print("\nTop confusion pairs (actual -> predicted):")
        print(misclassified.value_counts(["actual_category", "predicted_category"]).head(10).to_string())

    # ML PIPELINE SUMMARY
    #
    # 1. Load labeled transaction data       (train.csv; test.csv only if evaluating)
    # 2. Separate features and labels        (X = text, y = category)
    # 3. Learn vocabulary/IDF from TRAINING text only (TfidfVectorizer.fit)
    # 4. Convert training text into TF-IDF vectors (fit_transform)
    # 5. Train Logistic Regression on those vectors (learn class weights)
    # 6. Save the complete pipeline           (so it can be reused anywhere)
    # 7. [evaluate=True only] Transform unseen test text with the
    #    ALREADY-FITTED vectorizer and predict categories
    # 8. [evaluate=True only] Compare predictions with true labels
    # 9. [evaluate=True only] Analyze errors (confusion matrix, misclassified.csv)
    #
    # WHY THIS ORDER MATTERS: steps 3-5 may use ONLY training data; the test
    # set is first touched after all statistics are frozen. Any earlier
    # contact would leak test information into the model and inflate the
    # evaluation.
    return pipeline


def main() -> None:
    """Script entry point: the full experiment (train + save + evaluate)."""
    train_and_save_model(evaluate=True)


if __name__ == "__main__":
    main()
