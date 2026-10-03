"""SpendSense - main entry point.

WHAT THIS FILE IS
-----------------
A tiny command dispatcher: the single place you start the project from.
It contains NO machine-learning code itself - it only ROUTES to the real
implementations living in src/:

    main.py 1 | train+eval   -> train, evaluate on test.csv, save model
    main.py 2 | train        -> train + save ONLY (never opens test.csv)
    main.py 3 | test         -> grade the SAVED model on test.csv
    main.py 4 | predict      -> categorize typed text or a CSV file
    main.py 5 | unsupervised-train   -> train K-Means clustering model
    main.py 6 | unsupervised-analyze -> analyze discovered clusters
    main.py 7 | unsupervised-explore-k -> explore different K values
    main.py 8 | unsupervised-predict -> enter a description and get its cluster

WHY A DICT INSTEAD OF IF/ELIF (your "switch")
---------------------------------------------
Python has no classic switch statement. A dict mapping command -> function
is the idiomatic dispatch pattern: adding a command = one function + one
dict line, and the menu text lives right next to the routing table so they
cannot drift apart.

WHY THIS MATTERS FOR THE FUTURE
-------------------------------
A later UI, REST API, or scheduled job will not call training scripts
directly - it will call these same handler functions. Building the routing
habit now means "connecting" things later is adding entries to COMMANDS.

USAGE
-----
    python main.py                  (interactive menu - great in PyCharm)
    python main.py 1                (train + evaluate + save)
    python main.py 2                (train + save only)
    python main.py 3                (test the saved model)
    python main.py 4                (predict: type text or give a CSV path)
    python main.py 5                (train unsupervised clustering model)
    python main.py 6                (analyze discovered clusters)
    python main.py 7                (explore different K values)
    python main.py 8                (cluster inference for a new description)
    python main.py predict "Netflix subscription" "Shell gas station"
    python main.py predict --csv my_file.csv
"""

import sys

from src import predict, train_model
from src.unsupervised import clustering, analyze_clusters, explore_k, inference


# ---------------------------------------------------------------------------
# Command handlers - each takes leftover CLI arguments so the dispatcher can
# call every handler in exactly the same way.
# ---------------------------------------------------------------------------
def run_train_evaluate(_args: list[str]) -> None:
    """Mode 1 - the FULL experiment: train, evaluate, save.

    What happens inside (delegated to train_model.py - no copied code):
      1. Load train.csv
      2. Fit TF-IDF + Logistic Regression on training data only
      3. Save the pipeline to models/expense_category_model.joblib
      4. Evaluate on the unseen test set (accuracy / F1 / confusion matrix)
      5. Write error-analysis files
    """
    print("Mode 1: TRAIN + EVALUATE + SAVE")
    print("(training uses train.csv; evaluation uses the unseen test.csv)")
    train_model.train_and_save_model(evaluate=True)


def run_train_only(_args: list[str]) -> None:
    """Mode 2 - train and save WITHOUT any evaluation.

    WHY THIS MODE EXISTS
    --------------------
    Faster when you already know the baseline quality and just want a fresh
    model artifact (e.g. after changing the dataset). Guarantees by
    construction that test.csv is never opened - the test set stays
    completely untouched, which is the cleanest possible leakage protection.
    """
    print("Mode 2: TRAIN + SAVE ONLY")
    print("(test.csv will NOT be read - no evaluation will run)")
    train_model.train_and_save_model(evaluate=False)
    print("\nDone. Use mode 3 (test) when you want to grade this model,")
    print("or mode 4 (predict) to use it on new descriptions.")


def run_test_only(_args: list[str]) -> None:
    """Mode 3 - grade the ALREADY-SAVED model on test.csv.

    ML CAUTION: evaluating on test.csv repeatedly is fine; what would
    corrupt our generalization estimate is tweaking the model until the
    test scores look good - the test set would then be quietly influencing
    design decisions (test-set overfitting, a subtle form of leakage).
    """
    test_df = train_model.load_test_data()
    pipeline = predict.load_pipeline()

    print("Mode 3: TEST-ONLY (saved model evaluated on test.csv)")
    metrics = train_model.evaluate_model(pipeline, test_df)
    train_model.plot_confusion_matrix(metrics["y_test"], metrics["y_pred"])


def run_prediction(args: list[str]) -> None:
    """Mode 4 - categorize descriptions with the saved model.

    Input can be:
      - typed descriptions:   main.py predict "Netflix" "Shell gas"
      - a CSV file:           main.py predict --csv my_transactions.csv
      - nothing:              interactive prompt (type text or a CSV path)
    """
    predict.main(args)


def run_unsupervised_train(_args: list[str]) -> None:
    """Train the unsupervised clustering model."""
    print("Mode 5: TRAIN UNSUPERVISED CLUSTERING MODEL")
    print("(uses train.csv descriptions, no category labels)")
    clustering.run_clustering(k=10, random_state=42)
    print("\nDone. Use mode 6 to analyze clusters, or mode 8 to inference.")


def run_unsupervised_analyze(_args: list[str]) -> None:
    """Analyze discovered clusters."""
    print("Mode 6: ANALYZE CLUSTERS")
    analyze_clusters.analyze_clusters()
    print("\nDone. Check models/unsupervised/cluster_summary.csv for details.")


def run_unsupervised_explore_k(_args: list[str]) -> None:
    """Explore different K values."""
    print("Mode 7: EXPLORE DIFFERENT VALUES OF K")
    print("(sampled evaluation; may take a minute)")
    explore_k.explore_k(sample_size=5000, random_state=42)
    print("\nDone. Check models/unsupervised/k_exploration.json and k_elbow_plot.png.")


def run_unsupervised_inference(_args: list[str]) -> None:
    """Interactive inference for a new transaction description."""
    print("Mode 8: UNSUPERVISED CLUSTER INFERENCE")
    inference.run_interactive_inference()


# The routing table: command name -> handler function.
COMMANDS = {
    "1": run_train_evaluate,
    "train+eval": run_train_evaluate,
    "2": run_train_only,
    "train": run_train_only,
    "3": run_test_only,
    "test": run_test_only,
    "4": run_prediction,
    "predict": run_prediction,
    "5": run_unsupervised_train,
    "unsupervised-train": run_unsupervised_train,
    "6": run_unsupervised_analyze,
    "unsupervised-analyze": run_unsupervised_analyze,
    "7": run_unsupervised_explore_k,
    "unsupervised-explore-k": run_unsupervised_explore_k,
    "8": run_unsupervised_inference,
    "unsupervised-predict": run_unsupervised_inference,
}


def print_usage() -> None:
    """Show available commands, kept next to COMMANDS so they stay in sync."""
    print("Available commands:")
    print("  1 | train+eval               train + evaluate + save the model")
    print("  2 | train                    train + save only (no evaluation)")
    print("  3 | test                     evaluate the saved model on test.csv")
    print("  4 | predict [text | --csv f] categorize text or a CSV file")
    print("  5 | unsupervised-train       train K-Means clustering model")
    print("  6 | unsupervised-analyze     analyze discovered clusters")
    print("  7 | unsupervised-explore-k   explore different K values")
    print("  8 | unsupervised-predict     enter a description and get its cluster")
    print("  (run without arguments for the interactive menu)")


def interactive_menu() -> tuple[str, list[str]]:
    """Ask the user what to do when main.py starts without arguments.

    Ideal for PyCharm's Run button, which passes no command-line arguments.
    Returns (command, extra_args) in exactly the shape the dispatcher wants.
    """
    print("=" * 55)
    print("SpendSense")
    print("=" * 55)
    print("  1) train + evaluate + save model   (full experiment)")
    print("  2) train + save model only         (no evaluation)")
    print("  3) test the saved model            (on test.csv)")
    print("  4) predict                         (type text or give a CSV)")
    print("  5) train unsupervised clustering model")
    print("  6) analyze discovered clusters")
    print("  7) explore different values of K")
    print("  8) predict a cluster for a new description")
    print("  q) quit")

    while True:
        choice = input("Choose an option: ").strip().lower()
        if choice in {"1", "train+eval"}:
            return "1", []
        if choice in {"2", "train"}:
            return "2", []
        if choice in {"3", "test"}:
            return "3", []
        if choice in {"4", "predict"}:
            user_input = input(
                "Type a description, or a CSV path, or press Enter for samples: "
            ).strip()
            if user_input.lower().endswith(".csv"):
                return "4", ["--csv", user_input]
            return "4", [user_input] if user_input else []
        if choice in {"5", "unsupervised-train"}:
            return "5", []
        if choice in {"6", "unsupervised-analyze"}:
            return "6", []
        if choice in {"7", "unsupervised-explore-k"}:
            return "7", []
        if choice in {"8", "unsupervised-predict"}:
            return "8", []
        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)
        print("Please enter 1, 2, 3, 4, 5, 6, 7, 8, or q.")


def main() -> None:
    args = sys.argv[1:]

    # No arguments -> interactive menu. Otherwise the first word selects the
    # command and the rest are that command's arguments.
    if args:
        command, extra_args = args[0].lower(), args[1:]
    else:
        command, extra_args = interactive_menu()

    handler = COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command: {command!r}\n")
        print_usage()
        raise SystemExit(1)

    handler(extra_args)


if __name__ == "__main__":
    main()
